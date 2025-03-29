import os.path


def get_templates(text_prompt):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if 'simple' in text_prompt:
        simple_imagenet_template = ['a photo of a {}.']
        return simple_imagenet_template
    elif 'tip' in text_prompt:
        with open(os.path.join(current_dir, './prompt_tip.txt'), 'r', encoding='utf-8') as file:
            tip_imagenet_templates = [line.strip("'\"") for line in file.read().splitlines()]
        return tip_imagenet_templates
    elif 'vanilla' in text_prompt:
        return ['{}.']
    elif 'nice' in text_prompt:
        return ['The nice {}.']
    elif 'full' in text_prompt:
        with open(os.path.join(current_dir, './prompt.txt'), 'r', encoding='utf-8') as file:
            imagenet_templates = [line.strip("'\"") for line in file.read().splitlines()]
        return imagenet_templates
    else:
        raise NotImplementedError
