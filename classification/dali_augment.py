from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn



class NewBrightnessContrast:
    
    '''
    "brightness_contrast 亮度对比度
    '''
    def __init__(self,m):
        self.minvalue=0.
        self.maxvalue=0.4
        self.m=m
    def __call__(self,images):
        self.m= float(self.m)/30 *float(self.maxvalue-self.minvalue)+self.minvalue
        shift= fn.random.uniform(range=(0,self.m))
        con=fn.random.uniform(range=(0.6,1))
        return ops.BrightnessContrast(device = "gpu",brightness_shift=shift,contrast=con,contrast_center=100)(images)

    
class NewWater:
    def __init__(self,m):
        '''
        water
        只能用一个float
        '''
        self.m = m
        self.minvalue=0
        self.maxvalue=10

    def __call__(self,images):
        self.m= float(self.m)/30 *float(self.maxvalue-self.minvalue)+self.minvalue

        return ops.Water(device = "gpu",ampl_x=self.m,ampl_y=self.m)(images)


    
class NewShpere:
    def __init__(self,m):
        '''
        Shpere哈哈镜 就一个参数，mask,是否使用该变换
        '''
        self.m = m

    def __call__(self,images):
        p=fn.cast(fn.random.uniform(range=(0,1)),dtype=types.INT8)

        return ops.Sphere(device = "gpu",mask=p)(images)
    
class NewRotate:
    '''
    # 这个才是好的旋转
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        angle=fn.random.uniform(range=(0,1))*360
        return ops.Rotate(device = "gpu", angle =angle, interp_type = types.INTERP_LINEAR, fill_value = 0)(images)
    
class NewWarpAffine:
    '''
    # 这个不好改变参数,矩阵是特殊计算的，不会算
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        return ops.WarpAffine(device = "gpu", 
                              matrix = [1.0, 0.8, 0.0, 0.0, 1.2, 0.0],
                              interp_type = types.INTERP_LINEAR)(images)    

# twists colors of the image随机数设定
class NewHsv:
    '''
 
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        r= fn.cast(fn.random.uniform(range=(0,1))*100,dtype=types.FLOAT)
        b= fn.cast(fn.random.uniform(range=(0,1)),dtype=types.FLOAT)
        return ops.Hsv(device = "gpu",hue =r, saturation = b)(images)   

class NewJitte:
    '''

    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        p=fn.cast(fn.random.uniform(range=(0,1)),dtype=types.INT8)
        return ops.Jitter(device = "gpu",mask=p)(images) 
    
class NewPaste:
    '''
    放在空白背景上，随机数设定
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        x=fn.random.uniform(range=(0,1))
        y=fn.random.uniform(range=(0,1))
        return ops.Paste(device = "gpu",ratio = 2.,
                                   fill_value = (55, 155, 155),
                                    paste_x = x, paste_y = y)(images) 
    
class NewPaste:
    '''
    放在空白背景上，随机数设定
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        size = fn.cast(fn.random.uniform(range=(224,512)),dtype=types.INT8)

        return ops.Resize(device = "gpu", resize_shorter = 480)(images) 

class NewResize:
    '''
    400px long带短边的方法
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        short_size=fn.cast(fn.random.uniform(range=(224,512)),dtype=types.FLOAT)
        return ops.Resize(device = "gpu", resize_shorter =short_size)(images) 
    
class NewFlip:
    '''
    翻转
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        v = fn.cast(fn.random.uniform(range=(0,1)),dtype=types.INT32)
        h= fn.cast(fn.random.uniform(range=(0,1)),dtype=types.INT32)
        return ops.Flip(device = "gpu", vertical = v, horizontal = h)(images) 

class NewErase:
    '''
    翻转,暂时不知道怎么用的
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
        
        
        return ops.Erase(device = "gpu",anchor=[0.1,0.2], shape=[0.5, 0.6],
                                   normalized_anchor=True,
                                   normalized_shape=True, 
                                   axis_names='HW',
                                   fill_value= 100)(images) 

class NewSlice:
    '''
    翻转,暂时不知道怎么用的
    '''
    def __init__(self,m):
        pass
    def __call__(self,images):
    
        return ops.Slice(device = "gpu", rel_start=[0.3, 0.2], rel_shape=[0.5, 0.6], axis_names='HW')(images) 

    
augment_list=[NewSlice,NewErase,NewFlip,NewResize,NewPaste,NewPaste,NewJitte,NewHsv,NewWarpAffine,NewRotate,NewShpere,NewWater,NewBrightnessContrast]


import random
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list =augment_list

    def __call__(self, img):
        option = random.choices(self.augment_list, k=self.n)
        for op in option:
            img = op(self.m)(img)
        return img