import argparse

class Hparams:
    parser = argparse.ArgumentParser(description="SCMvL")
     
    # training scheme
    parser.add_argument("--logdir", dest="save", action="store", required=True, 
    			help="log directory")
    parser.add_argument("--lr", dest="lr", action="store", default=1e-3, type=float, 
    			help="learning rate")
    parser.add_argument("--epochs", dest="epochs", action="store", default=1000, type=int, 
    			help='')
    parser.add_argument("--batch_size", dest="batch_size", action="store", default=64, type=int, 
    			help='')
    parser.add_argument("--num_workers", dest="num_workers", action="store", default=8, type=int, 
    			help='')  
			
    # GNN varient models
    parser.add_argument("--model", dest="model", action="store", default="gcn", type=str,
			choices=["gcn", "graphsage", "gin"], help="Model architecture of the GNN Encoder")
    
    parser.add_argument("--layers", dest="layers", action="store", default=3, type=int, 
    			help='Number of graph convolution layers before each pooling')   			
    parser.add_argument("--feat_dim", dest="feat_dim", action="store", default=128, type=int, 
    			help='the dimension of node features in GNN')
    parser.add_argument("--loss", dest="loss", action="store", default="infonce", type=str)
    
    
    # Datasets		
    parser.add_argument("--dataset", dest="dataset", action="store", default="nci1", type=str,
			help="nci1, proteins, dd, enzymes, mutag, collab, imdb_multi, imdb_binary, reddit_multi, reddit_binary")
						
    # Augmentation pair for view generation		
    parser.add_argument("--augment_list", dest="augment_list", nargs="*", default=["node_attr_mask", "random_walk_subgraph"], type=str,
			choices=["node_dropping", "node_attr_mask", "edge_perturbation", "diffusion", "random_walk_subgraph"])
			
    parser.add_argument("--train_data_percent", dest="train_data_percent", action="store", default=1.0, type=float,
    			help="the fraction of pre-training samples")		


    """
    # Classification Task on Graphs
    parser.add_argument("--load", dest="load", action="store", help="Only when evaluating the best epoch")
    parser.add_argument("--classi_epochs", dest="classi_epochs", action="store", default=200, type=int)
    parser.add_argument("--runs", dest="runs", action="store", default=10, type=float, help="Number of experiments to conduct")
    """

class Classi_Hparams:
    parser = argparse.ArgumentParser(description="SCMvL for Graph Classifacation")
     
    # training scheme
    parser.add_argument("--logdir", dest="save", action="store", required=True, 
    			help="log directory")
    parser.add_argument("--lr", dest="lr", action="store", default=1e-3, type=float, 
    			help="learning rate")
    parser.add_argument("--epochs", dest="epochs", action="store", default=200, type=int, 
    			help='')
    parser.add_argument("--batch_size", dest="batch_size", action="store", default=64, type=int, 
    			help='')
    parser.add_argument("--num_workers", dest="num_workers", action="store", default=8, type=int, 
    			help='')
    			
    # GNN varient models
    parser.add_argument("--model", dest="model", action="store", default="gcn", type=str,
			choices=["gcn", "graphsage", "gin"], help="Model architecture of the GNN Encoder")
    
    parser.add_argument("--layers", dest="layers", action="store", default=3, type=int, 
    			help='Number of graph convolution layers before each pooling')   			
    parser.add_argument("--feat_dim", dest="feat_dim", action="store", default=128, type=int, 
    			help='the dimension of node features in GNN')
    
    
    # Datasets		
    parser.add_argument("--dataset", dest="dataset", action="store", default="nci1", type=str,
			help="nci1, proteins, dd, enzymes, mutag, collab, imdb_multi, imdb_binary, reddit_multi, reddit_binary")		
    parser.add_argument("--train_data_percent", dest="train_data_percent", action="store", default=1.0, type=float,
    			help="the fraction of pre-training samples")		

    # Classification Task on Graphs
    parser.add_argument("--load", dest="load", action="store", help="Only when evaluating the best epoch")
    parser.add_argument("--runs", dest="runs", action="store", default=10, type=float, help="Number of experiments to conduct")
								
  
"""
    ## vocabulary
    parser.add_argument('--vocab', default='iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")
    
"""
