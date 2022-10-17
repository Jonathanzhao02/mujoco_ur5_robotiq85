class Recorder():
    def __init__(self, obj_names, objective, gen_attrs):
        self.obj_names = obj_names
        self.objective = objective
        self.gen_attrs = gen_attrs
        self.cnt = 0
    
    def record(self, interface):
        img = interface.sim.render(255,255,camera_name='111')

        while img.sum() == 0:
            img = interface.sim.render(255,255,camera_name='111')

        feedback = interface.get_feedback()
        ee_pos = interface.get_xyz('EE')
        obj_pos = { obj: interface.get_xyz(obj) for obj in self.obj_names }

        state = {
            'img': img,
            'feedback': feedback,
            'ee_pos': ee_pos,
            'obj_pos': obj_pos,
            'timestep': self.cnt,
            'attributes': self.gen_attrs,
            'objective': self.objective
        } # add language templating, random initialization

        self.cnt += 1
