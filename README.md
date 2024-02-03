# Single-Shot Detection
* 함수 직관적 이해 돕기 위한 자료입니다.
  
## Dataset과 Dataloader 작성

### 1. 학습 및 검증용 이미지 데이터, 어노테이션 데이터 파일 경로 리스트 작성
#### 1-1. Make_datapath_list

Example:
###### VOCdevkit/VOC2012/
###### │
###### ├── Annotations/              # 어노테이션 XML 파일들이 저장된 폴더
###### │   ├── aaa.xml
###### │   ├── bbb.xml
###### │   └── ...
###### │
###### ├── JPEGImages/               # 이미지 JPG 파일들이 저장된 폴더
###### │   ├── aaa.jpg
###### │   ├── bbb.jpg
###### │   └── ...
###### │
###### └── ImageSets/Main/           # 데이터 세트 구분을 위한 텍스트 파일들이 있는 폴더
######     ├── train.txt
######     └── val.txt

###### train_img_list = ['.data/VOCdevkit/VOC2012/JPEGImages/aaa.jpg', ... ]
###### train_anno_list = ['.data/VOCdevkit/VOC2012/Annotations/bbb.xml', ... ]
###### val_img_list = ['.data/VOCdevkit/VOC2012/Annotations/ccc.xml', ... ]
###### val_anno_list = ['.data/VOCdevkit/VOC2012/Annotations/ddd.xml', ... ]

### 2. Dataset 작성
train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))


#### 2-1. transform은 이미지 자체에 적용되는 전처리 과정 => [[150,150,210,210,라벨], ... ]

#### 2-2. transform_anno는 객체 위치와 라벨 정보에 대한 전처리 과정 => [[xmin,ymin,xmax,ymax,label_index], ... ]

##### 2-2-1. XML 포맷의 annotation을 List로 변환
###### Anno_xml2list
######  <annotation>
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

######  <annotation>

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

#### 2-3. VOCDataset
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

### 3. DataLoader 작성
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)

## SSD 실행
### 1. SSD Configuration 설정 및 실행
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

#### 1-1. DBox(default box) 생성
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

#### 1-2. L2Norm()
∥x∥2 = root(∑ixi2) => 입력 x를 L2Norm으로 정규화한 뒤 초기 가중치 20인 weight와 곱한다. 
학습 과정에서 이 값들은 조정된다.
x의 값이 클루록 L2norm에 의해 더 많이 줄어들게 된다. 이는 신경망이 각 채널의 특징을 보다 균일하게 다루도록 도와준다.
(특히 object detection에서는 객체와 배경의 pixel값이 많이 다를거기에 이를 정규화 해주는 거다?)
​
 
​
### 2. MultiBoxLoss 손실함수
location_data = torch.Size([num_batch, 8732, 4]) --- 8732는 Default box 갯수, 4는 (xmin, ymin, xmax, ymax)
confidence_data = torch.Size([num_batch, 8732, 21]) --- 21은 class 갯수
dbox_list = torch.Size([8732,4)]

target = [[xmin, ymin, xmax, ymax, label_index], ... ] --- 실제 위치 정보

#### 2-1. location loss 계산 --- defaut box 위치와 실제 box 위치 계산
match(self.jaccard_thresh, truths, dbox_list.to(self.device), variance, labels, loc_t, conf_t_label, idx)
를 통해 실제 객체 위치와 default box의 위치 IOU가 0.5이상인 default box의 location을 loc_t에, label을 conf_t_label에 저장한다. 0.5보다 작으면 0으로 두고, 배경으로 저장한다.

##### 2-1-1. Smooth L1 fuction로 손실 계산 --- 단 물체 발견한 DBox의 coordinates만 계산(0인 배경은 계산 x)
SmoothL1Loss(x)=
{ 0.5 × x**2 if |x| < 1 }
{ |x| - 0.5 otherwise  }

Default box location과 Target box location 차이가 클 때는 L1, 차이가 작을 때는 L2 사용
L1 Loss: 차이가 0에 가까우면 Gradient Descent 시 불연속적 기울기로 너무 완만한 기울기 생성 가능성 높음
L2 Loss: 차이가 너무 크면 기울기 너무 커져 발산하거나 수렴 하지 못할 가능성 높음

따라서, 연속적인 기울기를 생성해 최적화 과정에서 안정성을 보장하기 위해 사용함.

#### 2-2. Confidence loss 계산 ---
loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')

Cross Entropy: H(y,p) = -∑Yklog(Pk) --- 예측 확률 분포와 실제 레이블 간의 차이를 측정하기 위함.
p = [0.1, 0.2, 0.7], y = [0,0,1]
H(y,p)=−(0×log(0.1)+0×log(0.2)+1×log(0.7))=−log(0.7)
실제 클래스를 확신하며, 이를 정확하게 예측하면 Cross Entropy 값은 0에 가까워짐.


Example:
미니배치 크기(num_batch): 2
디폴트 박스의 수(num_dbox): 4 (단순화를 위해 줄임)
클래스의 수(num_classes): 3 (클래스 0: 배경, 클래스 1: 고양이, 클래스 2: 개)

batch_conf = [
  [0.8, 0.1, 0.1],
  [0.2, 0.7, 0.1],
  [0.1, 0.1, 0.8],
  [0.6, 0.2, 0.2],
  [0.7, 0.2, 0.1],
  [0.1, 0.8, 0.1],
  [0.2, 0.2, 0.6],
  [0.5, 0.3, 0.2]
] --- 모델이 예측한 각 default box에 대한 클래스 신뢰도 담고 있는 텐서

conf_t_label = [0, 1, 2, 0, 0, 1, 2, 0] --- 각 default box에 대응하는 실제 클래스 레이블, location loss시 match함수에서 이미 생성.



### 3. SGD(Stochastic Gradient Descent) --- weight update, 손실함수 최소

def step(self, closure=None):
    """Performs a single optimization step (parameter update)."""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:  # 모든 파라미터 그룹에 대해 반복
        weight_decay = group['weight_decay']
        # 이전 업데이의 일부를 현재 업데이트에 반영
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']  # 학습률

        for p in group['params']:  # 현재 그룹 내의 모든 파라미터에 대해 반복
            if p.grad is None:
                continue
            d_p = p.grad.data  # 현재 파라미터의 그래디언트
            if weight_decay != 0:
                # θ_t+1 = θ_t - η⋅(∇θL(θ_t)+λθ_t) --- λ == weight_decay
                # ∇θL(θ_t): 손실함수 L에 대한 파라미터 θ_t의 그래디언트(미분값, 기울기)
                # 가중치의 미분을 빼주어 손실함수의 '골짜기'를 내려가 최솟값을 찾기 위함
                d_p.add_(p.data, alpha=weight_decay)  # Weight Decay 적용
                    
            if momentum != 0:  # 모멘텀 사용 시
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    # v_t+1 = μv_t + (1−τ)∇θL(θ_t)
                    # μ: momentum, 이전 v_t의 비율 결정, τ: dampening, 모멘텀에 그레디언트가 더해지는 비율을 감소
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)  # 모멘텀 업데이트
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)  # Nesterov 모멘텀 적용
                else:
                    d_p = buf
            # θ_t+1 = θ_t −ηv_t+1
            p.data.add_(d_p, alpha=-lr) 론

## SSD 추론
### 1. Detect(추론 시)

#### 1-1. Decode --- Deafult box -> Bounding Box 생성        
for i in range(num_batch):
        # loc와 DBox로 수정한 BBox [xmin, ymin, xmax, ymax] 를 구한다
        decoded_boxes = decode(loc_data[i], dbox_list)

loc_data에는 모델이 예측한 각 DBox의 Coordinate 담고 있음
dbox_list에는 원래 Default box의 Coordinate 담고 있음 

DBox = [cx, cy, width, height], prediced DBox = [Δcx, Δcy, Δwidth, Δheight]
c′x = cx + Δcx⋅w⋅scale_factor_1
c′y = cy + Δcx⋅h⋅scale_factor_1
w′ = w⋅exp(Δw)⋅scale_factor_2
h′ = h⋅exp(Δh)⋅scale_factor_2

​Xmin = c′x - w′/2
Ymin = c′y - h′/2
Xmax = c′x + w′/2
Ymax = c′y - h′/2

decoded_boxes = [[Xmin, Ymin, Xmax, Ymax], ... ]


decoded_boxes 중 confidence_threshold 미만인 BBox 제거

NMS(non-maximum suppression): 겹치는 BBox(IOU 계산) 중 confidence_score가 높은 BBox만 선택해 중복 제거

각 클래스 마다 신롸도가 가장 높은 상위 top_k만큼의 BBox만 남김.




















