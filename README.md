# Single-Shot Detection

## XML 포맷의 annotation을 List로 변환

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

학습 데이터와 검증 데이터 xml 포맷의 annotation을 tfrecord로 변환

model 생성

학습과 검증용 dataset을 생성하고, train 수행 



