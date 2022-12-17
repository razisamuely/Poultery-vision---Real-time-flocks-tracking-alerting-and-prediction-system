import xml.etree.ElementTree as ET


xml_input_path = "labeld_images_and_xml/2022-01-20-14-20-10.xml"
mytree = ET.parse(xml_input_path)
myroot = mytree.getroot()

for xmin, xmax in zip(myroot.iter('xmin'), myroot.iter('xmax')):
    xmin_t = x_shape - int(xmax.text)
    xmax_t = x_shape - int(xmin.text)

    xmax.text = str(int(xmax_t))
    xmin.text = str(int(xmin_t))

mytree.write(xml_output_path)