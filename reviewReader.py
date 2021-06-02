import glob, os, json, pickle
    

dir_path = 'C:\\Users\\Dino\\Desktop\\User-Review-Clustering\\app_reviews'
file = open('reviews.txt', 'wb')

for file2 in glob.glob(os.path.join(dir_path, '*.json')):
    with open(file2, 'r') as f:
        for element in f:
            data = json.loads(element)
            doc = (data['comment'] + "\n")
            pickle.dump(doc, file)
    f.close()

file.close()