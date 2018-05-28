
# # APD
python active_learning.py entropy model/config/cnn-globe-bn.yaml opt/config/sgld-mnist-2.yaml mnist --cuda --prefix apd --apd_gan opt/gan-config/mnist-wgan-gp-bs32.yaml
# # SGLD active
python active_learning.py entropy model/config/cnn-globe-bn.yaml opt/config/sgld-mnist-2.yaml mnist --cuda

# SGLD random
python active_learning.py random model/config/cnn-globe-bn.yaml opt/config/sgld-mnist-2.yaml mnist --cuda

# Deter active
python active_learning.py entropy model/config/cnn-globe-bn.yaml opt/config/sgld-mnist-2.yaml mnist --cuda --deter

# Deter random
python active_learning.py random model/config/cnn-globe-bn.yaml opt/config/sgld-mnist-2.yaml mnist --cuda --deter

# MC-Drop 
python active_learning.py entropy model/config/cnn-globe-bn.yaml opt/config/sgld-mnist-2.yaml mnist --cuda --deter --mc_dropout_passes 10000 --prefix mcd