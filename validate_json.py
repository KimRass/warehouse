import json
from jsonschema import validate


def validate_json(json_schema, json_to_validate, logger):
    try:
        validate(schema=json_schema, instance=json_to_validate)
    except Exception as e:
        logger.error(e)


def main():
    with open("/Users/jongbeom.kim/Desktop/workspace/Github/Work/data_mgmt/apps/b2b_projects/2022-AP-05_hase/hase_json_schema.json", mode="r") as f:
        json_schema = json.load(f)

    with open("/Users/jongbeom.kim/Desktop/workspace/Github/Work/data_mgmt/apps/b2b_projects/2022-AP-05_hase/samples_post/ko360020220521/ko360020220521_false.json", mode="r") as f:
        json_to_validate = json.load(f)

    validate_json(json_schema=json_schema, json_to_validate=json_to_validate)


if __name__ == "__main__":
    main()
