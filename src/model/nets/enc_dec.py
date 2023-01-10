import torch
from torch import nn 
from torch.distributions.normal import Normal

class EncDec(nn.Module):
    '''
    This connects the model encoder and decoder in one place.

    Args: 
        encoder(nn.Module): A model mapping batches of source domain to a
        batch of vectors
        decoder: Model mapping latent z to target domain
        sample z: whether to sample z or just take z= mu'''

    def __init_(
        self, 
        encoder,
        decoder,
        #...we can add further arguments here if we want it to be variational
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_size = self.decoder.input_shape[1]

    def forward(self, x):
        '''This is the encoder_decoder forward pass
        
        Args: 
            x = A batch of source domain data (batch x input_size)
            
        Returns: 
            The data after being passed through the encoder and the decoder    
        '''
        enc_out = self.encoder(x)
        output = self.decoder(enc_out)
        return output

    def load(self, weights_file):
        print("Trying to load model parameters from ", weights_file)
        self.load_state_dict(torch.load(weights_file))
        self.eval()
        print("Success!")
