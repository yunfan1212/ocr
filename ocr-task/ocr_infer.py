
from ocr_main import *
from tqdm import tqdm
def preprocess_input(image):
    image=image.astype(np.float32)
    image-=np.amin(image)
    image/=np.amax(image)
    return image


def dataprepare(image_path):
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image=cv2.resize(image,(128,64))
    image=preprocess_input(image)
    image=np.expand_dims(image,-1)
    image=np.transpose(image,(2,0,1)).astype(np.float32)
    image=torch.tensor(image,dtype=torch.float32).unsqueeze(0)
    return image



def test():
    args = get_args()
    myprocessor = MyProcessor(args.path)
    letters = myprocessor.get_vocab(args.label_vocab)
    args.num_classes = len(letters)
    args.max_len = myprocessor.get_max_len()

    features = [64, "M", 128, "M", 256, "M", 512, "M", 512]
    model = OCRModel(features=features, in_channels=1, out_features=32, num_classes=args.num_classes, num_layers=2)
    path = os.path.join(args.save_path, "model.pt")
    checkpoint_path=torch.load(path)
    model.load_state_dict(checkpoint_path)
    model.to(args.device)
    model.eval()


    label2id={k:v for k,v in enumerate(letters)}
    pred=[]
    with torch.no_grad():
        dfs=myprocessor.df
        for i in tqdm(range(len(dfs))):
            image_path=dfs.loc[i,"image_name"]
            path=image_path
            image=dataprepare(path)
            logits=model(image)

            pre_str=model.decode(logits,label2id,pad_ids=0)
            pred.append(pre_str)
    dfs["predict"]=pred
    dfs["t/f"]=dfs.loc[:,"class_name"]==dfs.loc[:,"predict"]
    print(dfs.loc[:,"t/f"].value_counts(normalize=True))
    dfs.to_csv('pre_result.csv', index = False)


if __name__ == '__main__':
    test()







