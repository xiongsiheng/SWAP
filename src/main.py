from models import Generator, Discriminator
from datasets import Dataset, load_dataset
from utlis import *
import gc
import sys
import argparse



parser = argparse.ArgumentParser()


parser.add_argument('--dataset', default='MATH')
parser.add_argument('--subset', default='algebra')
parser.add_argument('--prob_type', default='math')  # 'math', 'logical reasoning', 'coding'


parser.add_argument('--enable_DBM', type=bool, default=True)  # whether enable diversity-based modelling for generator
parser.add_argument('--use_meta_knowledge', type=bool, default=True)  # whether use meta-knowledge for discriminator
parser.add_argument('--visualize', type=bool, default=False)  # whether visualize the language model output


parser.add_argument('--output_dir', default='../results/test_MATH_algebra_llama3_8B')  # the output directory for inference results
parser.add_argument('--gen_model_id', default='../model_weights/MATH_algebra_Gen_llama3_8B/final')  # the path to the generator model
parser.add_argument('--sem_model_id', default='../model_weights/MATH_algebra_sem_equ_llama3_8B/final')  # the path to the semantical equivalence lora
parser.add_argument('--dis_model_id', default='../model_weights/MATH_algebra_Dis_llama3_8B/final')  # the path to the discriminator model

parser.add_argument('--max_steps', type=int, default=20)  # the maximum number of steps for reasoning
parser.add_argument('--num_rollouts', type=int, default=8)  # the number of rollouts for each problem
parser.add_argument('--num_generations', type=int, default=5)  # the number of generations for each step
parser.add_argument('--cmp_per_opt', type=int, default=1)  # the number of comparisons per option

parser.add_argument('--batch_size_gen', type=int, default=24)  # the batch size for generator
parser.add_argument('--batch_size_disc', type=int, default=12)  # the batch size for discriminator




args = parser.parse_args()





def SWAP(prob_type, dataset_test, output_dir, gen_model_id, sem_model_id, dis_model_id, meta_knowledge_path, max_steps=20, num_rollouts=8, num_generations=5, 
         cmp_per_opt=1, batch_size_gen=24, batch_size_disc=12, enable_DBM=True, use_meta_knowledge=True, visualize=False):
    '''
    Run the workflow of SWAP.

    Args:
        prob_type (str): The type of the problem.
        dataset_test (Dataset): The test dataset.
        output_dir (str): The output directory.
        gen_model_id (str): The path to the generator model.
        sem_model_id (str): The path to the semantical equivalence lora.
        dis_model_id (str): The path to the discriminator model.
        meta_knowledge_path (str): The path to the meta-knowledge.
        max_steps (int): The maximum number of steps for reasoning.
        num_rollouts (int): The number of rollouts for each problem.
        num_generations (int): The number of generations for each step.
        cmp_per_opt (int): The number of comparisons per option.
        batch_size_gen (int): The batch size for generator.
        batch_size_disc (int): The batch size for discriminator.
        enable_DBM (bool): Whether enable diversity-based modelling for generator.
        use_meta_knowledge (bool): Whether use meta-knowledge for discriminator.
        visualize (bool): Whether visualize the language model output.

    Returns:
        None
    '''
    # You can add more base models here.
    model_selection = 0
    model_name = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"][model_selection]

    num_generations = [1, 1] + [num_generations] * max_steps  # number of generations per step
    num_future_steps = [0, 0, max_steps] + [1, 0] * (max_steps//2)  # When planning for plan, we see all the future steps; 
                                                 # when planning for action, we see only the next step; 
                                                 # when planning for state, we see no future steps.

    for rollout_id in range(num_rollouts): 
        cnt = 0
        while cnt < max_steps:
            # Initialize Generator and perform inference
            agent_gen = Generator(gen_model_id, sem_model_id, model_name, enable_DBM=enable_DBM, prob_type=prob_type)
            force_termination = False if cnt < max_steps-1 else True
            flag_finish = agent_gen.inference(dataset_test, output_dir, str(rollout_id), batch_size_gen, num_generations[cnt], num_future_steps[cnt], 
                                              force_termination=force_termination, visualize=visualize)
            
            # Delete Generator instance to free memory
            del agent_gen
            gc.collect()

            if flag_finish:
                break
            
            # Initialize Discriminator and perform inference
            agent_disc = Discriminator(dis_model_id, model_name, use_meta_knwoledge=use_meta_knowledge, prob_type=prob_type)
            agent_disc.inference(output_dir, meta_knowledge_path, str(rollout_id), batch_size_disc, num_future_steps[cnt], cmp_per_opt, visualize=visualize)
            
            # Delete Discriminator instance to free memory
            del agent_disc
            gc.collect()

            cnt += 1
    
    # Perform final aggregation for all rollouts.
    agent_disc = Discriminator(dis_model_id, model_name, use_meta_knwoledge=use_meta_knowledge, prob_type=prob_type)
    agent_disc.inference(output_dir, meta_knowledge_path, "Agg", batch_size_disc, 0, cmp_per_opt, visualize=visualize, final_agg=True)



def build_dataset(args):
    '''
    Build the test dataset.

    Args:
        args (argparse.Namespace): The arguments.

    Returns:
        dataset_test: The test dataset.
        meta_knowledge_path: The path to the meta-knowledge.
    '''
    dataset = load_dataset("sxiong/SWAP", f"{args.dataset}_trajectory")
    print(dataset)

    dataset_filtered = []
    for sample in dataset['test']:
        if 'subset' in sample and args.subset != 'all':
            if args.subset != sample['subset']:
                continue
        del sample['trajectory'], sample['label']
        dataset_filtered.append(sample)

    dataset_filtered = dataset_filtered[:3]
    dataset_test = Dataset.from_list(dataset_filtered)
    
    meta_knowledge_path = f'../materials/{args.dataset}_{args.subset}_meta_knowledge'
    return dataset_test, meta_knowledge_path


if __name__ == '__main__':
    dataset_test, meta_knowledge_path = build_dataset(args)
    SWAP(args.prob_type, dataset_test, args.output_dir, args.gen_model_id, args.sem_model_id, args.dis_model_id, meta_knowledge_path, max_steps=args.max_steps, 
         num_rollouts=args.num_rollouts, num_generations=args.num_generations, cmp_per_opt=args.cmp_per_opt, batch_size_gen=args.batch_size_gen, 
         batch_size_disc=args.batch_size_disc, enable_DBM=args.enable_DBM, use_meta_knowledge=args.use_meta_knowledge, visualize=args.visualize)