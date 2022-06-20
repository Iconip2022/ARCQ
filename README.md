# This code will be updated after Oct.
# The quantization part is given here
# -------------------------------- auto quan---------------------------- #
# rcq.quan.utils
# find_modules_to_quantize() : Conv2d -> QuanConv2d , Linear QuanLinear
# QuanConv2d & QuanLinear rewrite in quan.func
# replace model 
#        args_quan = get_config(default_file='/code/mmclassification/rcq/config.yaml')
#        modules_to_replace = utils.find_modules_to_quantize(model, args_quan.quan,i_)
#        model = utils.replace_module_by_names(model,modules_to_replace)
