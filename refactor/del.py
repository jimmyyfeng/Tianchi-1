%run -i test_class.py  doTest --model_dir='checkpoints/luna2016_0701_03:29:34.pth' --img_dir='/mnt/7/0630_train_no_normalization/' --csv_file='new_train_1.csv'
%run -i collection/process.py check_nodule new_train_1.csv new_train_2.csv
 %run -i collection/process.py pcsv new_train_2.csv new_train_3.csv
%run -i collection/process.py check_nodule new_train_1.csv new_train_2.csv
%run -i collection/process.py pcsv new_train_2.csv new_train_3.csv
%run -i collection/cal_froc.py  main new_train_3.csv 1245 800
%hist
%hist -f del.py
