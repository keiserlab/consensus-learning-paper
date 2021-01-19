# # # !/bin/bash                         #-- what is the language of this shell

## individual nets
for fold in {0..3}
do 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "consensus" $fold 1 &> training_results/training_consensus_1_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "single net" "consensus" $fold 2 &> training_results/training_consensus_2_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "consensus" $fold 3 &> training_results/training_consensus_3_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "consensus" $fold 4 &> training_results/training_consensus_4_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "single net" "consensus" $fold 5 &> training_results/training_consensus_5_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold UG1 &> training_results/training_UG1_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold NP5 &> training_results/training_NP5_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "single net" "individual" $fold UG2 &> training_results/training_UG2_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold NP4 &> training_results/training_NP4_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold NP2 &> training_results/training_NP2_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "single net" "individual" $fold NP3 &> training_results/training_NP3_fold_${fold} &
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold NP1 &> training_results/training_NP1_fold_${fold} & 
	wait
done  

## individual nets - random labeler
for fold in {0..3}
do 
	CUDA_VISIBLE_DEVICES=1 python train.py "single net" "individual" $fold random0 &> training_results/training_random0_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold random1 &> training_results/training_random1_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold random2 &> training_results/training_random2_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=0 python train.py "single net" "individual" $fold random3 &> training_results/training_random3_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "single net" "individual" $fold random4 &> training_results/training_random4_fold_${fold} & 
	wait
done

## ensemble nets
for fold in {0..3}
do 
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "consensus" $fold 1 &> training_results/training_ensemble_consensus_1_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "consensus" $fold 2 &> training_results/training_ensemble_consensus_2_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "consensus" $fold 3 &> training_results/training_ensemble_consensus_3_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "consensus" $fold 4 &> training_results/training_ensemble_consensus_4_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "consensus" $fold 5 &> training_results/training_ensemble_consensus_5_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold UG1 &> training_results/training_ensemble_UG1_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "individual" $fold NP5 &> training_results/training_ensemble_NP5_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold UG2 &> training_results/training_ensemble_UG2_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "individual" $fold NP4 &> training_results/training_ensemble_NP4_fold_${fold} & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold NP2 &> training_results/training_ensemble_NP2_fold_${fold} & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "individual" $fold NP3 &> training_results/training_ensemble_NP3_fold_${fold} &
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold NP1 &> training_results/training_ensemble_NP1_fold_${fold} & 
	wait
done

CUDA_VISIBLE_DEVICES=1 python analysis.py 


## ensemble nets - random subnet
for fold in {0..3}
do 
	CUDA_VISIBLE_DEVICES=2 python train.py "ensemble" "consensus" $fold 1 "no amatuer" "random subnet" &> training_results/training_ensemble_consensus_1_fold_${fold}_random_subunit & 
	CUDA_VISIBLE_DEVICES=5 python train.py "ensemble" "consensus" $fold 2 "no amatuer" "random subnet" &> training_results/training_ensemble_consensus_2_fold_${fold}_random_subunit & 
	wait
	CUDA_VISIBLE_DEVICES=2 python train.py "ensemble" "consensus" $fold 3 "no amatuer" "random subnet" &> training_results/training_ensemble_consensus_3_fold_${fold}_random_subunit & 
	CUDA_VISIBLE_DEVICES=5 python train.py "ensemble" "consensus" $fold 4 "no amatuer" "random subnet" &> training_results/training_ensemble_consensus_4_fold_${fold}_random_subunit & 
	wait	
	CUDA_VISIBLE_DEVICES=2 python train.py "ensemble" "consensus" $fold 5 "no amatuer" "random subnet" &> training_results/training_ensemble_consensus_5_fold_${fold}_random_subunit & 
	CUDA_VISIBLE_DEVICES=5 python train.py "ensemble" "individual" $fold UG1 "no amatuer" "random subnet" &> training_results/training_ensemble_UG1_fold_${fold}_random_subunit & 
	wait
	CUDA_VISIBLE_DEVICES=2 python train.py "ensemble" "individual" $fold NP5 "no amatuer" "random subnet" &> training_results/training_ensemble_NP5_fold_${fold}_random_subunit & 
	CUDA_VISIBLE_DEVICES=5 python train.py "ensemble" "individual" $fold UG2 "no amatuer" "random subnet" &> training_results/training_ensemble_UG2_fold_${fold}_random_subunit & 
	wait	
	CUDA_VISIBLE_DEVICES=2 python train.py "ensemble" "individual" $fold NP4 "no amatuer" "random subnet" &> training_results/training_ensemble_NP4_fold_${fold}_random_subunit & 
	CUDA_VISIBLE_DEVICES=5 python train.py "ensemble" "individual" $fold NP2 "no amatuer" "random subnet" &> training_results/training_ensemble_NP2_fold_${fold}_random_subunit & 
	wait
	CUDA_VISIBLE_DEVICES=2 python train.py "ensemble" "individual" $fold NP3 "no amatuer" "random subnet" &> training_results/training_ensemble_NP3_fold_${fold}_random_subunit &
	CUDA_VISIBLE_DEVICES=5 python train.py "ensemble" "individual" $fold NP1 "no amatuer" "random subnet" &> training_results/training_ensemble_NP1_fold_${fold}_random_subunit & 
	wait
done
python analysis.py 


## ensemble nets - multiple subunits used
for fold in {0..3}
do 
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "consensus" $fold 1 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_consensus_1_fold_${fold}_multiple_random_subnets & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "consensus" $fold 2 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_consensus_2_fold_${fold}_multiple_random_subnets & 
	wait
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "consensus" $fold 3 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_consensus_3_fold_${fold}_multiple_random_subnets & 
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "consensus" $fold 4 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_consensus_4_fold_${fold}_multiple_random_subnets & 
	wait
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "consensus" $fold 5 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_consensus_5_fold_${fold}_multiple_random_subnets & 
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold NP5 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_NP5_fold_${fold}_multiple_random_subnets & 
	wait	
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold UG1 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_UG1_fold_${fold}_multiple_random_subnets & 
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "individual" $fold UG2 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_UG2_fold_${fold}_multiple_random_subnets & 
	wait	
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold NP4 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_NP4_fold_${fold}_multiple_random_subnets & 
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "individual" $fold NP2 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_NP2_fold_${fold}_multiple_random_subnets & 
	wait	
	CUDA_VISIBLE_DEVICES=0 python train.py "ensemble" "individual" $fold NP3 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_NP3_fold_${fold}_multiple_random_subnets &
	CUDA_VISIBLE_DEVICES=1 python train.py "ensemble" "individual" $fold NP1 "no amatuer" "no subunit" "multiple subnet" &> training_results/training_ensemble_NP1_fold_${fold}_multiple_random_subnets & 
	wait
done