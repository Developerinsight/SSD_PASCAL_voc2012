# Single-Shot Detection

## Dataset과 Dataloader 작성

### 학습 및 검증용 이미지 데이터, 어노테이션 데이터 파일 경로 리스트 작성
Make_datapath_list

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
Anno_xml2list

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

### SSD Configuration 설정 및 실행
ssd_cfg = {
    'num_classes': 21,  # 배경 클래스를 포함한 총 클래스 수
    'input_size': 300,  # 이미지의 입력 크기
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 출력할 DBox의 화면비의 종류
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 이미지 크기
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOX의 크기(최소)
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOX의 크기(최대)
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

net = SSD(phase='train', cfg=ssd_cfg)

#### DBox(default box) 생성
aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
feature map의 이미지 크기가 각 [38, 19, 10, 5, 3, 1]이라 하자.
steps는 [8, 16, 32, 64, 100, 300]이라 하자.
steps는 featrure map 상에서 default box가 위치할 영역의 크기를 결정 => step이 클수록 특징 맵은 더 큰 영역 커버하므로 큰 객체 탐지에 적합
min_size는 작은 정사각형의 Dbox 픽셀 크기
max_size는 큰 정사각형의 Dbox 픽셀 크기
=> 한 픽셀에 작은 Dbox, 큰 Dbox, 작은 Dbox의 종횡비 개수로 구성된다.
if aspect ratio == [2]: dbox ==4
else if aspect ratio == [2,3]: dbox ==6

총 dbox 갯수 = (38 * 38**2 * 4) + (19 * 19**2 * 6) + (10 * 10**2 * 6) + (5 * 5**2 * 6) + (3 * 3**2 * 4)
 + (1 * 1**2 * 4)

#### L2Norm()
∥x∥2 = root(∑ixi2) => 입력 x를 L2Norm으로 정규화한 뒤 초기 가중치 20인 weight와 곱한다. 
학습 과정에서 이 값들은 조정된다.
x의 값이 클루록 L2norm에 의해 더 많이 줄어들게 된다. 이는 신경망이 각 채널의 특징을 보다 균일하게 다루도록 도와준다.
(특히 object detection에서는 객체와 배경의 pixel값이 많이 다를거기에 이를 정규화 해주는 거다?)
​
 
​
### MultiBoxLoss 손실함수


### SGD

