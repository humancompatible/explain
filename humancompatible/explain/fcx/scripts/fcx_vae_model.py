import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class FCX_VAE(nn.Module):
    """
    Conditional Variational Autoencoder for generating feasible counterfactual explanations.

    This model encodes an input feature vector `x` concatenated with a target class label `c`
    into a latent representation, then decodes back to a counterfactual example. Supports
    multiple Monte Carlo draws and computes ELBO and validity conditioned outputs.

    Attributes:
        immutables (int): Number of initial immutable features (suffix features are mutable).
        encoded_size (int): Dimensionality of the latent code.
        data_size (int): Number of input features.
        encoded_categorical_feature_indexes (List[List[int]]): Index ranges for one-hot categorical features.
        encoded_continuous_feature_indexes (List[int]): Indices for continuous features.
        encoded_start_cat (int): Index where categorical features begin.
        encoder_mean (nn.Sequential): Network mapping input → latent mean.
        encoder_var (nn.Sequential): Network mapping input → latent variance.
        decoder_mean (nn.Sequential): Network mapping latent + label → reconstructed features.
    """
    def __init__(self, data_size, encoded_size, d,immutables=-2):
        """
        Initialize the FCX-VAE model.

        Args:
            data_size (int): Number of encoded input features (excluding label).
            encoded_size (int): Dimensionality of the latent space.
            d (DataLoader): Provides feature metadata (categorical splits, decoding).
            immutables (int, optional): Count of immutable feature columns at the end of `x`.
                                        Defaults to -2 (last two features are immutable).
        """
        super(FCX_VAE, self).__init__()
        
        self.immutables=immutables
        self.encoded_size = encoded_size
        self.data_size = data_size
        self.encoded_categorical_feature_indexes = d.get_data_params()[2]     
        
        self.encoded_continuous_feature_indexes=[]
        for i in range(self.data_size):
            valid=1
            for v in self.encoded_categorical_feature_indexes:
                if i in v:
                    valid=0
            if valid:
                self.encoded_continuous_feature_indexes.append(i)            

        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)

        # Plus 1 to the input encoding size and data size to incorporate the target class label        
        self.encoder_mean = nn.Sequential(
            nn.Linear( self.data_size+1, 20 ),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 20, 16 ),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 16, 14 ),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14,12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 12, self.encoded_size)
            )

        self.encoder_var = nn.Sequential(
            nn.Linear( self.data_size+1, 20 ),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 20, 16 ),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 16, 14 ),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14,12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 12, self.encoded_size),
            nn.Sigmoid()
            )

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.decoder_mean = nn.Sequential(
            nn.Linear( self.encoded_size+1, 12 ),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 12, 14 ),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 14, 16 ),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 16, 20 ),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 20, self.data_size),
            nn.Sigmoid()
            )
        
    def encoder(self, x):
        """
        Encode input (x + label) into latent mean and log‑variance.

        Args:
            x (torch.Tensor): Concatenated input and class label, shape (batch_size, data_size+1).

        Returns:
            mean (torch.Tensor): Latent means, shape (batch_size, encoded_size).
            logvar (torch.Tensor): Latent log-variances, shape (batch_size, encoded_size).
        """
        mean = self.encoder_mean(x)
        logvar = 0.5+ self.encoder_var(x)
        return mean, logvar

    def decoder(self, z):
        """
        Decode latent code back to feature space.

        Args:
            z (torch.Tensor): Concatenated latent code and label, shape (batch_size, encoded_size+1).

        Returns:
            torch.Tensor: Reconstructed features, shape (batch_size, data_size).
        """
        mean = self.decoder_mean(z)
        return mean
    
    def sample_latent_code(self, mean, logvar):
        """
        Perform the reparameterization trick to sample latent code.

        z = mean + sqrt(logvar) * eps, where eps ~ N(0, I).

        Args:
            mean (torch.Tensor): Latent means.
            logvar (torch.Tensor): Latent log-variances.

        Returns:
            torch.Tensor: Sampled latent code.
        """
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        """
        Compute the log-probability of x under N(mean, logvar).

        Args:
            x (torch.Tensor): Original or reconstructed features.
            mean (torch.Tensor): Latent means.
            logvar (torch.Tensor): Latent variances.
            raxis (int): Axis along which to sum log-likelihood.

        Returns:
            torch.Tensor: Log-likelihood per example.
        """
        return torch.sum( -.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar) ), axis=1)
                          
    def forward(self, x, c):
        """
        Generate multiple Monte Carlo counterfactual draws.

        Args:
            x (torch.Tensor): Original features, shape (batch_size, data_size).
            c (torch.Tensor): Target class label (0/1), shape (batch_size,).

        Returns:
            dict: {
                'em': encoder means,
                'ev': encoder variances,
                'z': list of sampled latent codes,
                'x_pred': list of reconstructed counterfactuals,
                'mc_samples': int number of MC draws
            }
        """
        c=c.view( c.shape[0], 1 )
        c=torch.tensor(c).float()        
        res={}
        mc_samples=50
        em, ev= self.encoder( torch.cat((x,c),1) )
        res['em'] =em
        res['ev'] =ev
        res['z'] =[]
        res['x_pred'] =[]
        res['mc_samples']=mc_samples
        for i in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred= self.decoder( torch.cat((z,c),1) )
            res['z'].append(z)
            res['x_pred'].append(x_pred)
    
        return res

    def compute_elbo(self, x, c, pred_model,ret=False):
        """
        Compute Evidence Lower Bound (ELBO) components for given inputs.

        If `ret` is True, also return de-normalized originals, reconstructions,
        predicted labels, and latent code for analysis.

        Args:
            x (torch.Tensor): Original features, may be truncated to mutable features if `ret`.
            c (torch.Tensor): Target labels, shape (batch_size,).
            pred_model (nn.Module): BlackBox classifier to evaluate validity.
            ret (bool): Whether to return extras for visualization.

        Returns:
            If ret=False:
                (log_px_z, kl_div, x, x_pred, cf_labels)
            If ret=True:
                (log_px_z, kl_div, x_orig, x_pred_mod, cf_labels, z_code)
        """
        c=torch.tensor(c).float()
        c=c.view( c.shape[0], 1 )    
        # Adult: -4 , #Census -7, law-2
        #
        #immutables=-4 #-2
        immutables=self.immutables
        if ret:
            x_copy=x.clone()
            x=x[:,:immutables] # Adult: -4 , #Census -7
        em, ev = self.encoder( torch.cat((x,c),1) )
        kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 

        z = self.sample_latent_code(em, ev)
        dm= self.decoder( torch.cat((z,c),1) )
        log_px_z = torch.tensor(0.0)
        
        x_pred= dm
        
        if ret:
            for i in range(1):
                x_copy2 = x_copy.clone()
                x_copy2[:,:immutables]=x_pred
                x_pred=x_copy2
                
        #print(x)
        #print(x_pred)
        if ret:
            return torch.mean(log_px_z), torch.mean(kl_divergence), x_copy, x_pred, torch.argmax( pred_model(x_pred), dim=1 ), z #em,ev#z
        else:
            return torch.mean(log_px_z), torch.mean(kl_divergence), x, x_pred, torch.argmax( pred_model(x_pred), dim=1 )

class AutoEncoder(nn.Module):
    
    def __init__(self, data_size, encoded_size, d):

        super(AutoEncoder, self).__init__()
        
        self.encoded_size = encoded_size
        self.data_size = data_size
        self.encoded_categorical_feature_indexes = d.get_data_params()[2]     

        self.encoded_continuous_feature_indexes=[]
        for i in range(self.data_size):
            valid=1
            for v in self.encoded_categorical_feature_indexes:
                if i in v:
                    valid=0
            if valid:
                self.encoded_continuous_feature_indexes.append(i)            

        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)
        
        self.encoder_mean = nn.Sequential(
            nn.Linear( self.data_size, 20 ),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 20, 16 ),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 16, 14 ),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14,12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 12, self.encoded_size)
        )

        self.encoder_var = nn.Sequential(
            nn.Linear( self.data_size, 20 ),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 20, 16 ),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 16, 14 ),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14,12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 12, self.encoded_size),
            nn.Sigmoid()
         )

        self.decoder_mean = nn.Sequential(
            nn.Linear( self.encoded_size, 12 ),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 12, 14 ),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 14, 16 ),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 16, 20 ),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear( 20, self.data_size),
            nn.Sigmoid()
            )
        
    def encoder(self, x):
        mean = self.encoder_mean(x)
        logvar = 0.05+ self.encoder_var(x)
        return mean, logvar

    def decoder(self, z):
        mean = self.decoder_mean(z)
        return mean
    
    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum( -.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar) ), axis=1)
                          
    def forward(self, x):        
        res={}
        mc_samples=50
        em, ev= self.encoder(x)
        res['em'] =em
        res['ev'] =ev
        res['z'] =[]
        res['x_pred'] =[]
        res['mc_samples']=mc_samples
        for i in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred= self.decoder(z)
            res['z'].append(z)
            res['x_pred'].append(x_pred)
    
        return res

