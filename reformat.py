import yaml
import os

dir_path = 'results'

for file in os.listdir(dir_path):
    if file.endswith('yaml'):
        print(os.path.join(dir_path, file))
        with open(os.path.join(dir_path, file)) as f:
            s = yaml.safe_load(f)
        print(s)
        if os.path.exists(os.path.join(dir_path, file)[:-5] + '.txt'):
            os.remove(os.path.join(dir_path, file))
        else:
            with open(os.path.join(dir_path, file)[:-5] + '.txt', 'w', encoding='utf-8') as f:
                for key in s:
                    f.write(str(key) + ' : ' + s[key] + '\n')
