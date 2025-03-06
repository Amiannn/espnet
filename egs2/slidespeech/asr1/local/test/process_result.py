import os

def read_csv(file):
    datas = []
    with open(file, 'r') as frs:
        for fr in frs:
            fr = fr.strip()
            data = fr.split(',')
            datas.append(data)
    return datas

def write_csv(file, datas):
    with open(file, 'w') as fws:
        for data in datas:
            fws.write(','.join(data) + '\n')

if __name__ == '__main__':
    result_folder = './exp/results'

    final_result_path  = os.path.join(result_folder, 'final_result.csv')
    domain_result_path = os.path.join(result_folder, 'domain_result.csv')

    