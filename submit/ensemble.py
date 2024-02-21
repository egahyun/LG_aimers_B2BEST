import pandas as pd
import numpy as np

import argparse

class Ensemble:
    def __init__(self, filenames:str, filepath:str):
        self.filenames = filenames
        self.output_list = []

        output_path = [filepath+filename+'.csv' for filename in filenames]
        self.output_frame = pd.read_csv(output_path[0]).drop('is_converted',axis=1)
        self.output_df = self.output_frame.copy()

        for path in output_path:
            self.output_list.append(pd.read_csv(path)['is_converted'].astype(int).to_list())
        for filename,output in zip(filenames,self.output_list):
            self.output_df[filename] = output


    def simple_weighted(self,weight:list):
        if not len(self.output_list)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.output_list[0]], [self.output_list[1]], axis=0)
        for i in range(2, len(self.output_list)):
            pred_arr = np.append(pred_arr, [self.output_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()


    def average_weighted(self):
        weight = [1/len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [pred*np.array(w) for pred, w in zip(self.output_list,weight)]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()


    def mixed(self):
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames)-1):
            pre_idx = self.filenames[idx]
            post_idx = self.filenames[idx+1]
            result[self.output_df[pre_idx]<1] = self.output_df.loc[self.output_df[pre_idx]<1,post_idx]
        return result.tolist()

def main(args):
    file_list = sum(args.ensemble_files, [])
    
    if len(file_list) < 2:
        raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")
    
    en = Ensemble(filenames = file_list,filepath=args.result_path)

    if args.ensemble_strategy == 'weighted':
        if args.ensemble_weight: 
            strategy_title = 'sw-'+'-'.join(map(str,*args.ensemble_weight)) #simple weighted
            result = en.simple_weighted(*args.ensemble_weight)
        else:
            strategy_title = 'aw' #average weighted
            result = en.average_weighted()
    elif args.ensemble_strategy == 'mixed':
        strategy_title = args.ensemble_strategy.lower() #mixed
        result = en.mixed()
    else:
        pass
    en.output_frame['is_converted'] = result
    output = en.output_frame.copy()
    output['is_converted'] = output['is_converted'].astype(bool)
    files_title = '-'.join(file_list)

    output.to_csv(f'{args.result_path}{strategy_title}-{files_title}.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg("--ensemble_files", nargs='+',required=True,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')
    arg('--ensemble_strategy', type=str, default='weighted',
        choices=['weighted','mixed'],
        help='optional: [mixed, weighted] 중 앙상블 전략을 선택해 주세요. (default="weighted")')
    arg('--ensemble_weight', nargs='+',default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help='optional: weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.')
    arg('--result_path',type=str, default='',
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다. (default:"")')
    args = parser.parse_args()
    main(args)