#!/usr/bin/python
# author zhanghan
# 2024年04月07日
from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()
