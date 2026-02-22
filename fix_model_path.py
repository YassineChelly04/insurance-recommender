
import json

def fix_model_path():
    file_path = r'c:\Users\yassi\OneDrive\Desktop\Data Overflow\dataoverflow-7-0.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the globals cell (cell with MODEL_PATH) and fix the filename
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'MODEL_PATH' in line:
                    new_source.append('MODEL_PATH = "model.joblib"\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Fixed MODEL_PATH to model.joblib in dataoverflow-7-0.ipynb")

if __name__ == "__main__":
    fix_model_path()
