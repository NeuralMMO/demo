from pdb import set_trace as T

import nmmo

from config import bases, scale
import tasks


import nmmo

from scripted import baselines

class RLlib:
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters

   Configure NUM_GPUS and NUM_WORKERS for your hardware
   Note that EVALUATION_NUM_WORKERS cores are reserved for evaluation
   and one additional core is reserved for the driver process.
   Therefore set NUM_WORKERS <= cores - EVALUATION_NUM_WORKERS - 1
   '''

   @property
   def MODEL(self):
      return self.__class__.__name__

   @property
   def TRAIN_BATCH_SIZE(self):
      return 64 * 256 * self.NUM_WORKERS

   #Checkpointing. Resume will load the latest trial, e.g. to continue training
   #Restore (overrides resume) will force load a specific checkpoint (e.g. for rendering)
   EXPERIMENT_DIR          = 'experiments'
   RESUME                  = False

   RESTORE                 = True
   RESTORE_ID              = 'Baseline' #Experiment name suffix
   RESTORE_CHECKPOINT      = 1000

   #Hardware and debug
   NUM_GPUS_PER_WORKER     = 0
   LOCAL_MODE              = False
   LOG_LEVEL               = 1

   #Training and evaluation settings
   EVALUATION_INTERVAL     = 1
   EVALUATION_PARALLEL     = True
   TRAINING_ITERATIONS     = 1000
   KEEP_CHECKPOINTS_NUM    = 3
   CHECKPOINT_FREQ         = 1
   LSTM_BPTT_HORIZON       = 16
   NUM_SGD_ITER            = 1

   #Model
   SCRIPTED                = None
   NPOLICIES               = 1

   #Reward
   COOPERATIVE             = False
   TEAM_SPIRIT             = 0.0


class Small(RLlib, nmmo.config.Small):
   '''Small scale Neural MMO training setting

   Features up to 64 concurrent agents and 32 concurrent NPCs,
   64 x 64 maps (excluding the border), and 128 timestep horizons'''
   
   
   #Memory/Batch Scale
   ROLLOUT_FRAGMENT_LENGTH = 128
   SGD_MINIBATCH_SIZE      = 128
 
   #Horizon
   TRAIN_HORIZON           = 128
   EVALUATION_HORIZON      = 128


class Medium(RLlib, nmmo.config.Medium):
   '''Medium scale Neural MMO training setting

   Features up to 256 concurrent agents and 128 concurrent NPCs,
   128 x 128 maps (excluding the border), and 1024 timestep horizons'''
 
   #Memory/Batch Scale
   ROLLOUT_FRAGMENT_LENGTH = 256
   SGD_MINIBATCH_SIZE      = 128
 
   #Horizon
   TRAIN_HORIZON           = 1024
   EVALUATION_HORIZON      = 1024


class Large(RLlib, nmmo.config.Large):
   '''Large scale Neural MMO training setting

   Features up to 2048 concurrent agents and 1024 concurrent NPCs,
   1024 x 1024 maps (excluding the border), and 8192 timestep horizons'''
 
   #Memory/Batch Scale
   ROLLOUT_FRAGMENT_LENGTH = 32
   SGD_MINIBATCH_SIZE      = 128

   #Horizon
   TRAIN_HORIZON           = 8192
   EVALUATION_HORIZON      = 8192

class Medium(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems):
    '''Config for NMMO default environment with concurrent spawns'''
    TASKS                   = tasks.All

    # Load 1000 epoch pretrained model
    RESTORE                 = True
    RESTORE_ID              = '870d'

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT
    pass

class CompatibleMedium(Medium):
    EMULATE_FLAT_OBS       = True
    EMULATE_FLAT_ATN       = True
    EMULATE_CONST_POP      = True
    EMULATE_CONST_HORIZON  = True

class Debug(scale.Debug, bases.Small, nmmo.config.AllGameSystems):
   '''Debug Neural MMO training setting

   A version of the Smallsetting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''
   EMULATE_FLAT_OBS       = True
   EMULATE_FLAT_ATN       = True
   EMULATE_CONST_POP      = True
   EMULATE_CONST_HORIZON  = True

   TASKS                   = tasks.All

   RESTORE                 = False

   TRAINING_ITERATIONS     = 2

   SGD_MINIBATCH_SIZE      = 100
   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2

class QMix(scale.Debug, bases.Small):
   '''Debug Neural MMO training setting

   A version of the Smallsetting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''

   @property
   def SPAWN(self):
       return self.SPAWN_CONCURRENT

   NENT = 1
 
   TASKS                   = tasks.All

   RESTORE                 = False

   TRAINING_ITERATIONS     = 2

   SGD_MINIBATCH_SIZE      = 100
   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2


