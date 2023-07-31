import argparse
import json
import os.path as osp

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fail_info", type=str, help="path of fail_info.json")
    parser.add_argument("data", type=str, help="root path of dataset")
    args = parser.parse_args()

    with open(args.fail_info, "r") as f:
        fail_records = json.load(f)

    with open(osp.join(args.data, "annotation/sample_data.json"), "r") as f:
        sample_data_records = json.load(f)

    print(f"Num Fail: {len(fail_records)}")

    for fail_info in fail_records:
        timestamp = fail_info["timestamp"]
        sample_data = None
        for sd_record in sample_data_records:
            if sd_record["timestamp"] == timestamp:
                sample_data = sd_record
        if sample_data is None:
            print(f"[WARN]: There is no corresponding sample data!!, timestamp: {timestamp}")
            continue
        filename = sample_data["filename"]
        img = cv2.imread(osp.join(args.data, filename))
        for fp_result in fail_info.get("fp", []):
            img = cv2.putText(
                img,
                text=f"[FP-Est]: {fp_result['est']}",
                org=(30, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
            )
            img = cv2.putText(
                img,
                text=f"[FP-GT]: {fp_result['gt']}",
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255),
            )

        for fn_result in fail_info.get("fn", []):
            img = cv2.putText(
                img,
                text=f"[FN-GT]: {fn_result['gt']}",
                org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255),
            )

        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27:
            break


if __name__ == "__main__":
    main()
