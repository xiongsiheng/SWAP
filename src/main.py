from models import Generator, Discriminator
from datasets import Dataset
from utlis import *
import os
import gc



def SWAP(dataset_test, output_dir, gen_model_id, sem_model_id, disc_model_id, max_steps=20):
    num_generations = 3  # search tree width
    cmp_per_opt = 5 # number of comparisons per option
    num_future_steps = [1, 0, 20] + [1, 0] * 10  # When planning for plan, we see all the future steps; 
                                                 # when planning for action, we see only the next step; 
                                                 # when planning for state, we see no future steps.
    batch_size_gen = 24
    batch_size_disc = 12

    cnt = 0
    while cnt < max_steps:
        # Initialize Generator and perform inference
        agent_gen = Generator(gen_model_id, sem_model_id, enable_DBM=True, show_prompt_only=False, prob_type='math')
        force_termination = False if cnt < max_steps-1 else True
        flag_finish = agent_gen.inference(dataset_test, output_dir, batch_size_gen, num_generations, num_future_steps[cnt], force_termination=force_termination, 
                            visualize=False)
        
        # Delete Generator instance to free memory
        del agent_gen
        gc.collect()

        if flag_finish:
            break
        
        # Initialize Discriminator and perform inference
        agent_disc = Discriminator(disc_model_id, enable_meta_knwoledge=False, show_prompt_only=False, prob_type='math')
        agent_disc.inference(output_dir, batch_size_disc, num_future_steps[cnt], cmp_per_opt, visualize=False)
        
        # Delete Discriminator instance to free memory
        del agent_disc
        gc.collect()

        cnt += 1
    


def build_dataset(dataset_dir):
    file_ls = [f'{dataset_dir}/{file}' for file in os.listdir(dataset_dir)]
    file_ls = sorted(file_ls)
    data_dict_test = obtain_data_dict(file_ls)
    dataset = Dataset.from_dict(data_dict_test)
    return dataset


if __name__ == '__main__':
    dataset_test = build_dataset("../dataset/MATH/test/algebra")
    output_dir = '../results/test_llama3_8B_SFT_algebra'
    gen_model_id = "../model_weights/MATH_Gen/final"
    sem_model_id = "../model_weights/MATH_Sem_equ/final"
    disc_model_id = '../model_weights/MATH_Discriminator/final'
    
    SWAP(dataset_test, output_dir, gen_model_id, sem_model_id, disc_model_id, max_steps=20)