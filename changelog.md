# Code refactoring
Add travis to follow code modification

## Test directory
Add test code to make sure the program will always work even in the future with big modification or code refactoring. 

## Doc directory
Put file to run test code and example code. 

## IO directory
Move `load_NDskl` from skel to io directory. Change the output by generate :
- dictionary of specs
- dataframe of critical point
- dataframe of filament
- dataframe of points contains in each filament

Add log and warn to `save_fits` function

Simplify `load_image`, and add metadata dictionnary (only pixel xy ratio information for now). 

Add `save_image` to save an np.array into a tiff file.

Add `save_skeleton` and `load_skeleton` to save and load skeleton information in HF5 file. 

## Image directory
Move `import_im` function from io directory to image directory as `z_project` function. Combine with `proj_around_max` function. Made some simplification of projection moreover when it concerns the full stack. 

## Geometry
Create `Skeleton` class, with different dataframe to save each information (critical points, filaments and points). This class contains different method:
- `remove_lonely_cp` : remove isolated critical point connected to any filament
- `remove_free_filament` : remove filament which in one side are not connected to an other filament. At the end we obtain a closed skeleton. 
- `create_binary_image` : create an np.array from the skeleton structure. 

## Segmentation
Always use `mask` np.array as filament=0 and background=1. 

## Utils
Move `fil_mask_int` to `Skeleton` class as `create_binary_image`.
Move `img_dilation` to image.py as `dilation` function. 
Remove `fil_dilation` which to the same as `dilation` function.
Remove `xyz_from_array` which can be done with `x,y,z = np.where(image>0)`. 
Remove `create_df_from_skeleton` since we load NDskl skeleton as `Skeleton` object. 

## Analysis
Add `general_analysis` function, which measure signal mean and std of background and skeleton signal. 



