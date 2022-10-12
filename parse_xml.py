from xml.etree.ElementTree import XMLParser, ElementTree, Element, indent
from pathlib import Path
import os
import sys

'''
This program reads an XML tree, finds elements with tag {tag}, then replaces them with {__n} occurrences
of elements with tag {__tag} and attributes defined by the original element's attributes

e.g. with {tag=stl_insert}

<stl_insert __n='3' __tag='mesh' name='box{i:03d}' mesh='box{i:03d}' euler='1.57 0 0' />

becomes

<mesh name='box000' mesh='box000' euler='1.57 0 0' />
<mesh name='box001' mesh='box001' euler='1.57 0 0' />
<mesh name='box002' mesh='box002' euler='1.57 0 0' />
'''
def parse_xml(f_xml, tag, new_f_xml):
    if not f_xml.exists():
        raise Exception("XML file does not exist")

    tree = ElementTree()
    tree.parse(str(f_xml))

    tag_els = tree.findall(f'.//{tag}/..')

    for el in tag_els:
        els = el.findall(f'./{tag}')

        for e in els:
            attrib = e.attrib.copy()
            tag = attrib.pop('__tag')
            n = int(attrib.pop('__n', '64'))

            if tag is None:
                raise Exception('No __tag attribute found in element')

            el.remove(e)

            for i in range(n):
                a = attrib.copy()

                for key,val in a.items():
                    a[key] = str.format(val, i=i)
                
                new_e = Element(tag, a)
                el.append(new_e)

    indent(tree, '    ')
    tree.write(str(new_f_xml))
