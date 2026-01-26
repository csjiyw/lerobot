./rclone lsf huawei-cloud:openloong-apps-prod-private/data-collector-svc   --dirs-only --max-depth 2 | grep -E "^([^/]+/){1}9f22b6b23ed64ced8fc8074d454e5672/$"
./rclone size huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/9f22b6b23ed64ced8fc8074d454e5672/
./rclone copy   huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/9f22b6b23ed64ced8fc8074d454e5672   ./data/9f22b6b23ed64ced8fc8074d454e5672   --progress --transfers 8 --checkers 16

537e9a6f62034a19aad056fe21b53fbb



data_id=537e9a6f62034a19aad056fe21b53fbb

export data_id=537e9a6f62034a19aad056fe21b53fbb

./rclone lsf huawei-cloud:openloong-apps-prod-private/data-collector-svc \
  --dirs-only --max-depth 2 \
  | grep -E "^([^/]+/){1}${data_id}/$"


./rclone size \
  huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/${data_id}/

./rclone copy \
  huawei-cloud:openloong-apps-prod-private/data-collector-svc/align/${data_id} \
  ./data/${data_id} \
  --progress --transfers 8 --checkers 16
