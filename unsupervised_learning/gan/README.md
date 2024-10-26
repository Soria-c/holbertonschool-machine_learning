# GANs

## Resources

* [GoodFellow et al. (2014\)](/rltoken/pb7tygh2Jl7jAmKPVdmAgg "GoodFellow et al. (2014)")
* [Arjovsky et al. (2017\)](/rltoken/smDUOqM_gUQJT1RwKLudZg "Arjovsky et al. (2017)")
* [Gulrajani et al. (2017\)](/rltoken/5zHXXBinoXdWcxQfQndBCQ "Gulrajani et al. (2017)")
* [Introduction to the Wasserstein distance](/rltoken/RV-8dwSTWvefxQkZyZAGBA "Introduction to the Wasserstein distance")
* [This person does not exist](/rltoken/8fcZNa4l51xS2jNR-LOiBQ "This person does not exist")

## Basics about GANs

* The description contains a crash course about GANs. The fundamentals ideas behind GANs are exposed as fast as possible.
* Task 0 : the class `Simple_GAN` : as they were introduced, GANs are a game played by two adversary players.
* Task 1 : the class `WGAN_clip` : later it appeared that in fact the two players can collaborate.
* Task 2 : the class `WGANGP` *: a more natural version of the latter, which outperforms `Simple`*`GAN`s and `WGAN_clip`s in higher dimensions, as will be illustrated in the main program.
* Task 3 : convolutional generators and discriminators : to work with pictures we need to use convolutional neural networks.
* Task 4 : our own face generator : we will use a `WGAN_GP` model to produce faces of persons that don’t exist.
* Appendix : at the end of task 4 there is a short digression explaining why GANs are superior to the older generators of fake pictures that were based on PCA.

## Tasks

### 0\. Simple GAN






None

```
def spheric_generator(nb_points, dim) :
      u=tf.random.normal(shape=(nb_points, dim))
      return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u),axis=[1])+10**-8),[nb_points,1])
  
  def fully_connected_GenDiscr(gen_shape, real_examples, latent_type="normal" ) :
      
      #   Latent generator   
      if latent_type   == "uniform" :
          latent_generator  =  lambda k : tf.random.uniform(shape=(k, gen_shape[0]))
      elif latent_type == "normal" :
          latent_generator  =  lambda k : tf.random.normal(shape=(k, gen_shape[0])) 
      elif latent_type == "spheric" :
          latent_generator  = lambda k : spheric_generator(k,gen_shape[0]) 
      
      #   Generator  
      inputs     = keras.Input(shape=( gen_shape[0] , ))
      hidden     = keras.layers.Dense( gen_shape[1] , activation = 'tanh'    )(inputs)
      for i in range(2,len(gen_shape)-1) :
          hidden = keras.layers.Dense( gen_shape[i] , activation = 'tanh'    )(hidden)
      outputs    = keras.layers.Dense( gen_shape[-1], activation = 'sigmoid' )(hidden)
      generator  = keras.Model(inputs, outputs, name="generator")
      
      #   Discriminator     
      inputs        = keras.Input(shape=( gen_shape[-1], ))
      hidden        = keras.layers.Dense( gen_shape[-2],   activation = 'tanh' )(inputs)
      for i in range(2,len(gen_shape)-1) :
          hidden    = keras.layers.Dense( gen_shape[-1*i], activation = 'tanh' )(hidden)
      outputs       = keras.layers.Dense( 1 ,              activation = 'tanh' )(hidden)
      discriminator = keras.Model(inputs, outputs, name="discriminator")
      
      return generator, discriminator, latent_generator
  
```

### 1\. Wasserstein GANs






* fill in the `generator_loss` function in the `__init__` method
* fill in the `discriminator_loss` function in the `__init__` method
* fill in the `train_step` method



```
import tensorflow as tf
  from tensorflow import keras
  import numpy as np
  import matplotlib.pyplot as plt
  
  class WGAN_clip(keras.Model) :
      
      def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
          super().__init__()                         # run the __init__ of keras.Model first. 
          self.latent_generator = latent_generator
          self.real_examples    = real_examples
          self.generator        = generator
          self.discriminator    = discriminator
          self.batch_size       = batch_size
          self.disc_iter        = disc_iter
          
          self.learning_rate    = learning_rate
          self.beta_1=.5                               # standard value, but can be changed if necessary
          self.beta_2=.9                               # standard value, but can be changed if necessary
          
          # define the generator loss and optimizer:
          self.generator.loss      = lambda x : pass           # <----- new !
          self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
          self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
          
          # define the discriminator loss and optimizer:
          self.discriminator.loss      = lambda x,y : pass   # <----- new !
          self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
          self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
         
      
      # generator of real samples of size batch_size
      def get_fake_sample(self, size=None, training=False):
          if not size :
              size= self.batch_size
          return self.generator(self.latent_generator(size), training=training)
  
      # generator of fake samples of size batch_size
      def get_real_sample(self, size=None):
          if not size :
              size= self.batch_size
          sorted_indices = tf.range(tf.shape(self.real_examples)[0])
          random_indices  = tf.random.shuffle(sorted_indices)[:size]
          return tf.gather(self.real_examples, random_indices)
          
               
      # overloading train_step()    
      def train_step(self,useless_argument):
          pass
          #for _ in range(self.disc_iter) :
              
              # compute the loss for the discriminator in a tape watching the discriminator's weights
                  # get a real sample
                  # get a fake sample
                  # compute the loss discr_loss of the discriminator on real and fake samples
              # apply gradient descent once to the discriminator
              
              # clip the weights (of the discriminator) between -1 and 1    # <----- new !
  
          # compute the loss for the generator in a tape watching the generator's weights 
              # get a fake sample 
              # compute the loss gen_loss of the generator on this sample
          # apply gradient descent to the generator
          
          
          
          # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
  
```

### 2\. Wasserstein GANs with gradient penalty






* fill in the `generator_loss` function in the `__init__` method
* fill in the `discriminator_loss` function in the `__init__` method
* fill in the `train_step` method



```
class WGAN_GP(keras.Model) :    
      def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005,lambda_gp=10):
          super().__init__()                         # run the __init__ of keras.Model first. 
          self.latent_generator = latent_generator
          self.real_examples    = real_examples
          self.generator        = generator
          self.discriminator    = discriminator
          self.batch_size       = batch_size
          self.disc_iter        = disc_iter
          
          self.learning_rate    = learning_rate
          self.beta_1=.3                              # standard value, but can be changed if necessary
          self.beta_2=.9                              # standard value, but can be changed if necessary
          
          self.lambda_gp        = lambda_gp                                # <---- New !
          self.dims = self.real_examples.shape                             # <---- New !
          self.len_dims=tf.size(self.dims)                                 # <---- New !
          self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')   # <---- New !
          self.scal_shape=self.dims.as_list()                              # <---- New !
          self.scal_shape[0]=self.batch_size                               # <---- New !
          for i in range(1,self.len_dims):                                 # <---- New !
              self.scal_shape[i]=1                                         # <---- New !
          self.scal_shape=tf.convert_to_tensor(self.scal_shape)            # <---- New !
          
          # define the generator loss and optimizer:
          self.generator.loss  = lambda x : pass                                  # <---- to be filled in                 
          self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
          self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
          
          # define the discriminator loss and optimizer:
          self.discriminator.loss      = lambda x , y :pass                         # <---- to be filled in  
          self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
          self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
  
      # generator of real samples of size batch_size
      def get_fake_sample(self, size=None, training=False):
          if not size :
              size= self.batch_size
          return self.generator(self.latent_generator(size), training=training)
  
      # generator of fake samples of size batch_size
      def get_real_sample(self, size=None):
          if not size :
              size= self.batch_size
          sorted_indices = tf.range(tf.shape(self.real_examples)[0])
          random_indices  = tf.random.shuffle(sorted_indices)[:size]
          return tf.gather(self.real_examples, random_indices)
      
      # generator of interpolating samples of size batch_size              # <---- New !
      def get_interpolated_sample(self,real_sample,fake_sample):
          u = tf.random.uniform(self.scal_shape)
          v=tf.ones(self.scal_shape)-u
          return u*real_sample+v*fake_sample
      
      # computing the gradient penalty                                     # <---- New !
      def gradient_penalty(self,interpolated_sample):
          with tf.GradientTape() as gp_tape:
                  gp_tape.watch(interpolated_sample)
                  pred = self.discriminator(interpolated_sample, training=True)
          grads = gp_tape.gradient(pred, [interpolated_sample])[0]
          norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
          return tf.reduce_mean((norm - 1.0) ** 2)      
       
           # overloading train_step()    
      def train_step(self,useless_argument):
          pass
          #for _ in range(self.disc_iter) :
              
              # compute the penalized loss for the discriminator in a tape watching the discriminator's weights
              
                  # get a real sample
                  # get a fake sample
                  # get the interpolated sample (between real and fake computed above)
                                  
                  # compute the old loss discr_loss of the discriminator on real and fake samples        
                  # compute the gradient penalty gp
                  # compute the sum new_discr_loss = discr_loss + self.lambda_gp * gp                     
                                  
              # apply gradient descent with respect to new_discr_loss once to the discriminator 
  
          # compute the loss for the generator in a tape watching the generator's weights 
          
              # get a fake sample 
              # compute the loss gen_loss of the generator on this sample
              
          # apply gradient descent to the discriminator (gp is the gradient penalty)
          
          # return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp":gp}        
  
```

### 3\. Generating faces

Here we will convince ourselves that gans can be trained into generators of faces. In order to do so, we first need a large enough sample of real faces.

* For the generator, the input data will have shape (16\)
* For the discriminator, the input data will have shape (16,16,1\)
* Use a hyperbolic tangent activation function “tanh” (even in Dense layers)
* In every Conv2D, the padding\=“same”
* Returns: the concatenated output of the generator and discriminator



```
# load the pictures
  import matplotlib.pyplot as plt 
  array_of_pictures=np.load("faces/small_res_faces_10000.npy")
  array_of_pictures=array_of_pictures.astype("float32")/255
  
  fig,axes=plt.subplots(10,10,figsize=(10,10))
  fig.suptitle("real faces")
  for i in range(100) :
      axes[i//10,i%10].imshow(array_of_pictures[i,:,:])
      axes[i//10,i%10].axis("off")
  plt.show()
  
```

### 4\. Our own "This person does not exist" : Playing with a pre\-trained model

The developers of this project have trained a  `WGAN_GP`  model just as you did in the last task, but for a larger number of epoch (150\), of a larger number of steps (200\) with a small learning rate (.0001\) and stored 

* The weights of the generator in a file  [generator.h5](https://drive.google.com/uc?export=download&id=1KiRQKhaTPxI9UeGKkdyaMpAoixfhRQf8)
* The weights of the discriminator in a file  [discriminator.h5](https://drive.google.com/uc?export=download&id=19S7JmeMSjKtx_DU5K-p8zpZU5NJLAUyt)



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  import numpy as np
  import tensorflow as tf
  import random
  import matplotlib.pyplot as plt
  
  WGAN_GP = __import__('4-wgan_gp').WGAN_GP
  convolutional_GenDiscr = __import__('3-generate_faces').convolutional_GenDiscr
  
  
  ## Regulating the seed
  
  def set_seeds(seed) :
      random.seed(seed)
      np.random.seed(seed)
      tf.random.set_seed(seed)
  
  def normal_generator(nb_points, dim) :
      return tf.random.normal(shape=(nb_points, dim))
  
  
  ## Load the pictures from small_res_faces_10000
  
  array_of_pictures=np.load("small_res_faces_10000.npy")
  array_of_pictures=array_of_pictures.astype("float32")/255
  
  ## Center and Normalize the data
  
  mean_face=array_of_pictures.mean(axis=0)
  centered_array = array_of_pictures - mean_face
  multiplier=np.max(np.abs(array_of_pictures),axis=0)
  normalized_array = centered_array/multiplier
  
  ## An example of a large real sample
  
  latent_generator  =  lambda k : normal_generator(k, 16)
  
  real_ex = tf.convert_to_tensor(normalized_array,dtype="float32")
  
  ## Visualize the fake faces : Code to generate the first picture
  
  def recover(normalized) :
      return normalized*multiplier+mean_face
  
  def plot_400(G) :
      Y=G.get_fake_sample(400)
      Z=G.discriminator(Y)
      Znu=Z.numpy()
      Ynu=Y.numpy()
      inds=np.argsort(Znu[:,0])
      H=Ynu[inds,:,:,0]
      fig,axes=plt.subplots(20,20,figsize=(20,20))
      fig.subplots_adjust(top=0.95)
      fig.suptitle("fake faces")
      for i in range(400) :
          axes[i//20,i%20].imshow(recover(H[i,:,:]))
          axes[i//20,i%20].axis("off")
      plt.show()
  
  ## LET'S GO
  
  set_seeds(0)
  generator,discriminator = convolutional_GenDiscr()
  G=WGAN_GP(generator,discriminator,latent_generator, real_ex,batch_size=200, disc_iter=2,learning_rate=.001)
  G.replace_weights("generator.h5","discriminator.h5")
  plot_400(G)
  
  $ ./0-main.py
  
```

