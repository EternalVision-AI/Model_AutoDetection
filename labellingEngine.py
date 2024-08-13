
import os

def makeLabelling(dirpath, jpgfname, imgsizechannel, labellists):
    """

    :param dirpath:
    :param jpgfname:        jpg filename
    :param xmlfname:        xml filename
    :param imgsizechannel:  (h, w, c)
    :param labellists:      [[objname, xmin, ymin, xmax, ymax], ...]
    :return:
    """
    res_str = """
    <annotation verified="yes">
        <folder>JPEGImages</folder>
        <filename>""" + str(jpgfname) + """</filename>
        <path>""" + dirpath + """</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>""" + str(imgsizechannel[1]) + """</width>
            <height>""" + str(imgsizechannel[0]) + """</height>
            <depth>""" + str(imgsizechannel[2]) + """</depth>
        </size>
        <segmented>0</segmented>"""
    for lbl in labellists:
        objname, xmin, ymin, xmax, ymax = lbl
        res_str = res_str + """
        <object>
            <name>""" + str(objname) + """</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>""" + str(xmin) + """</xmin>
                <ymin>""" + str(ymin) + """</ymin>
                <xmax>""" + str(xmax) + """</xmax>
                <ymax>""" + str(ymax) + """</ymax>
            </bndbox>
        </object>
        """
    res_str = res_str + """
    </annotation>
    """
    return res_str


