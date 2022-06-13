import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-w', type=int, help='Window size')
    parser.add_argument('-r', type=int, help='run')

    parser.add_argument('-nh', type=int, help='number-of-head')
    parser.add_argument('-md', type=int, help='mlp-dimension')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nl', type=int, help='num_layers')
    args = parser.parse_args()
    model_name = args.m
    window_size = args.w
    run = args.r
    num_heads=args.nh
    mlp_dim=args.md
    num_layers=args.nl
    hidden_size=args.ed

    path = '/geoinfo_vol1/zhao2/proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run).join('_'+str(num_heads)).join('_'+str(mlp_dim)).join('_'+str(hidden_size)).join('_'+str(num_layers))
    new_path = '/geoinfo_vol1/zhao2/proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)
    os.system('mv '+path+' '+new_path)