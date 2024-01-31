# Single-Shot Detection

## Dataset과 Dataloader 작성

### 학습 및 검증용 이미지 데이터, 어노테이션 데이터 파일 경로 리스트 작성
Example:
VOCdevkit/VOC2012/
│
├── Annotations/              # 어노테이션 XML 파일들이 저장된 폴더
│   ├── aaa.xml
│   ├── bbb.xml
│   └── ...
│
├── JPEGImages/               # 이미지 JPG 파일들이 저장된 폴더
│   ├── aaa.jpg
│   ├── bbb.jpg
│   └── ...
│
└── ImageSets/Main/           # 데이터 세트 구분을 위한 텍스트 파일들이 있는 폴더
    ├── train.txt
    └── val.txt

train_img_list = ['.data/VOCdevkit/VOC2012/JPEGImages/aaa.jpg', ... ]
train_anno_list = ['.data/VOCdevkit/VOC2012/Annotations/bbb.xml', ... ]
val_img_list = ['.data/VOCdevkit/VOC2012/Annotations/ccc.xml', ... ]
val_anno_list = ['.data/VOCdevkit/VOC2012/Annotations/ddd.xml', ... ]

### Dataset 작성
train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))


#### transform은 이미지 자체에 적용되는 전처리 과정 => [[150,150,210,210,라벨], ... ]

#### transform_anno는 객체 위치와 라벨 정보에 대한 전처리 과정 => [[xmin,ymin,xmax,ymax,label_index], ... ]
##### XML 포맷의 annotation을 List로 변환
Example:

<annotation>
    <folder>VOC2012</folder>
    <filename>image_1.jpg</filename>
    <size>
        <width>800</width>
        <height>600</height>
        <depth>3</depth>
    </size>
    <object>
        <name>cat</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>400</xmax>
            <ymax>500</ymax>
        </bndbox>
    </object>
    <object>
        <name>dog</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>450</xmin>
            <ymin>300</ymin>
            <xmax>650</xmax>
            <ymax>550</ymax>
        </bndbox>
    </object>
</annotation>

이미지 내 모든 물체의 annotation을 이 리스트에 저장
ret = []

xml = ET.parse(xml_path).getroot()

이미지 내에 있는 물체(object) 수만큼 반복
for obj in xml.iter('object'):
    
    한 물체의 annotation을 저장하는 리스트
    bndbox = []
    
    물체 이름
    name = obj.find('name').text.lower().strip()
    
    바운딩 박스 정보
    bbox = obj.find('bndbox')
    
    annotation의 xmin, ymin, xmax, ymax 취득 후 정규화
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    
    for pt in (pts):
        voc는 원점이 (1,1)이므로 1을 빼서 (0,0)으로 한다
        cur_pixel = int(bbox.find(pt).text) -1

        x 방향의 경우 폭으로 나눈다
        if pt == 'xmin' or pt == 'xmax':
            cur_pixel /= width
        y 방향의 경우 높이로 나눈다
        else:
            cur_pixel /= height

        bndbox.append(cur_pixel)

    어노테이션의 클래스명 index를 취득하여 추가
    label_idx = self.classes.index(name)
    bndbox.append(label_idx)

    ret += [bndbox]

[[xmin, ymin, xmax, ymaxm label_ind], ... ]
return np.array(ret) 

#### VOCDataset
img = cv2.imread(image_file_path)
height, width, channels = img.shape

xml형식의 transform된 annotation 정보 list에 저장
anno_list = transform_anno(anno_file_path, width, height)

전처리 실시
img, boxes, labels = transform(img, phase, anno_list[:,:4], anno_list[:,4])

BGR -> RGB
img = torch.from_numpy(img[:,:,(2,1,0)]).permutate(2,0,1)

gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
=> Example: [[100, 200, 300, 400, 0],
             [150, 250, 350, 450, 1], ... ]

return img, gt, height, width

### DataLoader 작성
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
