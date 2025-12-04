✔ ~/Documents/urban_model/amd/improved $ python MSI_cropped.py  

Loading MSI data from /home/sac/Documents/urban_model/amd/MSI.tif...

MSI array shape: (4, 3425, 3936)



Resampling label to match MSI dimensions...

Label shape: (3425, 3936)



--- Extracting Patches ---

Total patches extracted: 3119

Training patches: 2495

Validation patches: 624



Using device: cuda



--- Training Model with Early Stopping and CutMix ---

Using Focal Loss for class imbalance

Traceback (most recent call last):

 File "/home/sac/Documents/urban_model/amd/improved/MSI_cropped.py", line 1141, in <module>

   trained_model = train_model(

       model, train_loader, val_loader,

   ...<4 lines>...

       early_stopping_patience=30

   )

 File "/home/sac/Documents/urban_model/amd/improved/MSI_cropped.py", line 1015, in train_model

   loss = criterion(outputs, masks)

 File "/home/sac/miniconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl

   return self._call_impl(*args, **kwargs)

          ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

 File "/home/sac/miniconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl

   return forward_call(*args, **kwargs)

 File "/home/sac/Documents/urban_model/amd/improved/MSI_cropped.py", line 875, in forward

   bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')

 File "/home/sac/miniconda3/lib/python3.13/site-packages/torch/nn/functional.py", line 3560, in binary_cross_entropy

   raise ValueError(

   ...<2 lines>...

   )

ValueError: Using a target size (torch.Size([8, 1, 128, 128])) that is different to the input size (torch.Size([8, 1, 42, 42])) is deprecated. Please ensure they have the same size.
