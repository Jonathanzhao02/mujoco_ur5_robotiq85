from xml.etree.ElementTree import XMLParser, ElementTree, Element, indent
from pathlib import Path
import os
import sys
import random

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

COLORS = {
    'red': [1, 0, 0],
    'green': [0, 1, 0],
    'blue': [0, 0, 1],
    'yellow': [1, 1, 0],
    'pink': [1, 0, 1],
    'cyan': [0, 1, 1],
    'white': [1, 1, 1],
    'black': [0, 0, 0]
}

SIZES = {
    'small': [0.7, 0.9],
    'medium': [0.9, 1.1],
    'large': [1.1, 1.3]
}

def truthy(s):
    if s.lower() not in ['false', 'f', '']:
        return True
    else:
        return False

def gen_color():
    color_word = random.choice(list(COLORS.keys()))
    new_rgb = [0, 0, 0]
    rgb = COLORS[color_word]
    
    for i in range(3):
        new_rgb[i] = min(1, max(0, rgb[i] + random.uniform(-0.1, 0.1)))
    
    return color_word, str.format('{r:.3f} {g:.3f} {b:.3f} 1', r=new_rgb[0], g=new_rgb[1], b=new_rgb[2])

def gen_size():
    size_word = random.choice(list(SIZES.keys()))
    size = random.uniform(*SIZES[size_word])

    return size_word, size

'''
This method reads an XML tree, finds elements with tag {tag}, then replaces them with {__n} occurrences
of elements with tag {__tag} and attributes defined by the original element's attributes

e.g. with {tag=stl_insert}

<stl_insert __n='3' __tag='mesh' name='box{i:03d}' mesh='box{i:03d}' euler='1.57 0 0' />

becomes

<mesh name='box000' mesh='box000' euler='1.57 0 0' />
<mesh name='box001' mesh='box001' euler='1.57 0 0' />
<mesh name='box002' mesh='box002' euler='1.57 0 0' />
'''
def parse_xml(f_xml, template_tag, new_f_xml):
    if not f_xml.exists():
        raise Exception("XML file does not exist")

    tree = ElementTree()
    tree.parse(str(f_xml))

    tag_els = tree.findall(f'.//{template_tag}/..')

    gen_colors = {}
    gen_sizes = {}
    gen_scales = {}

    for el in tag_els:
        els = el.findall(f'./{template_tag}')

        for e in els:
            attrib = e.attrib.copy()
            __tag = attrib.pop('__tag')
            __n = int(attrib.pop('__n', '64'))
            __id = attrib.pop('__id', ''.join([random.choice(ALPHABET) for _ in range(16)]))
            __rand_color = attrib.pop('__rand_color', 'false')
            __rand_size = attrib.pop('__rand_size', 'false')
            __rand_scale = attrib.pop('__rand_scale', 'false')
            
            color_word = None
            color = None
            size_word = None
            size = None
            scale_word = None
            scale = None

            if __tag is None:
                raise Exception('No __tag attribute found in element')
            
            if truthy(__rand_color):
                color_word, color = gen_color()
                gen_colors[__id] = [color_word, color]
            
            if truthy(__rand_size):
                size = attrib.pop('size')
                size_word, size_factor = gen_size()
                size = ' '.join([str.format('{size:.5f}', size=float(i) * size_factor) for i in size.split(' ')])
                gen_sizes[__id] = [size_word, size_factor]
            
            if truthy(__rand_scale):
                scale = attrib.pop('scale')
                scale_word, scale_factor = gen_size()
                scale = ' '.join([str.format('{scale:.5f}', scale=float(i) * scale_factor) for i in scale.split(' ')])
                gen_scales[__id] = [scale_word, scale_factor]
            
            el.remove(e)

            for i in range(__n):
                a = attrib.copy()

                for key,val in a.items():
                    a[key] = str.format(val, i=i)
                
                if color is not None:
                    a['rgba'] = color

                if size is not None:
                    a['size'] = size
                
                if scale is not None:
                    a['scale'] = scale
                
                new_e = Element(__tag, a)
                el.append(new_e)

    indent(tree, '    ')
    tree.write(str(new_f_xml))

    return gen_colors, gen_sizes, gen_scales
