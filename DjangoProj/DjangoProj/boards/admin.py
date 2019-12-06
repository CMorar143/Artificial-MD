from django.contrib import admin
from .models import Patient, Doctor, Examination

admin.site.register(Patient)
admin.site.register(Doctor)
admin.site.register(Examination)

# Register your models here.
