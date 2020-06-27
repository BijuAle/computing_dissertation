from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from random import randint
from django.contrib import messages
import numpy as np
import math

def show_home(request):
    return render(request, 'home.htm')

def show_load_c(request):
    data, tf = load_dataset(request)
    return render(request, 'load_c.html', get_dataset_preview(data, tf))

def load_dataset(request):
    data = None
    tf = None
    valid_extensions = ['.npy']
    
    if request.method == 'POST':
        file = request.FILES['file']
        fs = FileSystemStorage()
        # fs.save(file.name, file)
        ext = os.path.splitext(file.name)[1]
        if not ext.lower() in valid_extensions:
            messages.info(request, 'Invalid File. Please upload a valid NumPY file.')
            return data, tf
        else:
            # loading the file
            data = np.load(file)
            tf = float(request.POST.get('training_fraction'))
            messages.info(request, 'Dataset: \"' +file.name + '\" loaded Sucessfully!')
    return data, tf

def get_dataset_preview(data, tf):
    model = {}
    if data is None and tf is None:
        return model

    # setting the fraction of data which should be in the training set
    fraction_training = tf

    # spliting the data
    training, testing = splitdata_train_test(data, fraction_training)

    # printing the key values
    model = {
        'size': len(data),
        'n_training_fraction': fraction_training,
        'n_training_set': len(training),
        'n_testing_set': len(testing)
    }

    # print an example
    for name, value in zip(data.dtype.names, data[randint(0, len(data))]):
        # Make model characters Template-Friendly
        name = name.replace('-', '_')
        model[name] = value
    return model


def splitdata_train_test(data, fraction_training):
    # randomizing data set order
    np.random.seed(0)
    np.random.shuffle(data)

    # find split point
    training_rows = math.floor(len(data) * fraction_training)
    training_set = data[:training_rows]
    testing_set = data[training_rows:len(data)]

    return (training_set, testing_set)