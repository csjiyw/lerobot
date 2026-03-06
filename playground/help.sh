obs://openloong-apps-prod-private/ali_a2d_batch3

./rclone lsf huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/   --dirs-only > folders.txt

./rclone lsf huawei-cloud:openloong-apps-prod-private/ali_a2d_batch3  --dirs-only > ali_a2d_batch3.txt

./rclone lsf huawei-cloud:openloong-apps-prod-private/ali_a2d_batch3 \
  --dirs-only --max-depth 2 \
  | grep -E "^([^/]+/){1}fea69b9a6d0440ada940ac432da0c48c/$"


./rclone lsf huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/01a81bc5a7df4551a903bb4d342c6eb7

./rclone lsf huawei-cloud:openloong-apps-prod-private/01a81bc5a7df4551a903bb4d342c6eb7


./rclone copy \
  huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/6542d39bbbd14f5cab153aea90fd8af2/573e33f3f9fd4e5183b05f43e240ba51 \
  /mnt/sdc/data/6542d39bbbd14f5cab153aea90fd8af2/573e33f3f9fd4e5183b05f43e240ba51 \
  --progress --transfers 8 --checkers 16

/mnt/sdc/data/0044f0e441f35a7eaa39889dd41be09b/000f47dc526252e497ea2da889809c6b.h5
./rclone copy \
  huawei-cloud:openloong-apps-prod-private/ali_a2d_batch7/0044f0e441f35a7eaa39889dd41be09b/000f47dc526252e497ea2da889809c6b.h5 \
  /mnt/fastdisk/tmp_download/0044f0e441f35a7eaa39889dd41be09b \
  --progress --transfers 8 --checkers 16



huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/1da29204155f4843a1257cbfbd001fab/269200e82f9447979aee8c9bb9b80fec/
./rclone copy \
  huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/01a81bc5a7df4551a903bb4d342c6eb7 \
  /mnt/sdc/tmp_download/01a81bc5a7df4551a903bb4d342c6eb7 \
  --progress --transfers 4 --checkers 16


./rclone copy \
  huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/${task_id}/${episode_id} \
  ./data/${task_id}/${episode_id} \
  --progress --transfers 8 --checkers 16

fea69b9a6d0440ada940ac432da0c48c

python ginie1_align2lerobot_v30.py \
--input /mnt/sdc/data/6542d39bbbd14f5cab153aea90fd8af2/573e33f3f9fd4e5183b05f43e240ba51  \
--output /mnt/sdc/output  \
--repo_id 6542d39bbbd14f5cab153aea90fd8af2

python ginie1_align2lerobot_v30.py \
--input /mnt/sdc/tmp_download/0044f0e441f35a7eaa39889dd41be09b  \
--output /mnt/sdc/output/0044f0e441f35a7eaa39889dd41be09b  \
--repo_id 0044f0e441f35a7eaa39889dd41be09b


  /mnt/sdc/tmp_download/1da29204155f4843a1257cbfbd001fab 

SVT_LOG=1 python ginie1_align2lerobot_v30.py \
--input /mnt/sdc/tmp_download/1da29204155f4843a1257cbfbd001fab   \
--output /mnt/sdc/output/1da29204155f4843a1257cbfbd001fab  \
--repo_id 1da29204155f4843a1257cbfbd001fab  \
--task abc_is good
--workers 1

SVT_LOG=1 python ginie1_align2lerobot_v30.jpeg.py \
--input /mnt/sdc/tmp_download/78726502a65b5227aa8bb4d5353e9303   \
--output /mnt/sdc/output/78726502a65b5227aa8bb4d5353e9303  \
--repo_id 78726502a65b5227aa8bb4d5353e9303  \
--task abc_is good  \
--workers 10


SVT_LOG=1  python download_and_process.2.py


SVT_LOG=1 python gr2_align2lerobot_v30.jpeg.py \
--input /mnt/sdc/tmp_download/01a81bc5a7df4551a903bb4d342c6eb7   \
--output /mnt/sdc/test/01a81bc5a7df4551a903bb4d342c6eb7  \
--repo_id 01a81bc5a7df4551a903bb4d342c6eb7  \
--task abc_is good  \
--workers 1


HF_HOME=/mnt/sdc/hf_cache TMPDIR=/mnt/sdb/tmp  python download_and_process.3.gr2.py