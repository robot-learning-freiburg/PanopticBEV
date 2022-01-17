CUDA_VISIBLE_DEVICES={CUDA GPU IDs} \
python3 -m torch.distributed.launch --nproc_per_node={NUM GPUs} --master_addr={IP ADDR} --master_port={PORT NUM} train_panoptic_bev.py \
                                    --run_name={NAME OF THE RUN} \
                                    --project_root_dir={PATH OF THE panoptic_bev FOLDER} \
                                    --seam_root_dir={PATH OF THE NUSCENES SEAMLESS DIRECTORY} \
                                    --dataset_root_dir={PATH OF THE NUSCENES DATASET} \
                                    --mode=train \
                                    --train_dataset=nuScenes \
                                    --val_dataset=nuScenes \
                                    --config=nuscenes.ini
