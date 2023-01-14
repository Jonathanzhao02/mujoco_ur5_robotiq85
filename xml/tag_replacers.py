import random

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

class TagReplacer:
    def __init__(self, tag, generator_fn):
        self.tag = tag
        self.generate = generator_fn
        self.value = None
        self.descriptor = None
    
    def next(self, attr):
        if truthy(attr.pop(self.tag, 'false')):
            self.descriptor, self.value = self.generate()
        else:
            self.descriptor, self.value = None, None
    
    def modify_attr(self, attr):
        raise NotImplementedError

class ColorTagReplacer(TagReplacer):
    def __init__(self):
        TagReplacer.__init__(
            self,
            '__rand_color',
            gen_color,
        )
    
    def modify_attr(self, attr, parent):
        if self.value is not None:
            attr['rgba'] = self.value

class ScaleTagReplacer(TagReplacer):
    def __init__(self):
        TagReplacer.__init__(
            self,
            '__rand_scale',
            gen_size,
        )
    
    def modify_attr(self, attr, parent):
        if self.value is not None:
            scale = parent.get('scale')
            attr['scale'] = ' '.join([str.format('{scale:.5f}', scale=float(i) * self.value) for i in scale.split(' ')])

class SizeTagReplacer(TagReplacer):
    def __init__(self):
        TagReplacer.__init__(
            self,
            '__rand_size',
            gen_size,
        )
    
    def modify_attr(self, attr, parent):
        if self.value is not None:
            size = parent.get('size')
            attr['size'] = ' '.join([str.format('{size:.5f}', size=float(i) * self.value) for i in size.split(' ')])
