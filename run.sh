rm -rf ./output
mkdir ./output
export profile="wiki10k" && python -X pycache_prefic=./cache train_acm.py > ./output/wiki10k.log
export profile="twitter" && python -X pycache_prefic=./cache train_acm.py  > ./output/twitter.log
export profile="ppi" && python -X pycache_prefic=./cache train_acm.py  > ./output/ppi.log
export profile="dblp" && python -X pycache_prefic=./cache train_acm.py  > ./output/dblp.log
export profile="blogcatalog" && python -X pycache_prefic=./cache train_acm.py  > ./output/blogcatalog.log