from django.contrib import admin
from .models import Patient, Doctor, Examination

admin.site.register(Patient)
admin.site.register(Doctor)
admin.site.register(Examination)

# Register your models here.
class ExamAdmin(admin.ModelAdmin):
	fields = ('MALE', 'gender', 'FEMALE')

	radio_fields = {'gender': admin.VERTICAL}

	# fields = ('MALE', 'gender', 'FEMALE')

	radio_fields = {'chest_pain': admin.VERTICAL}
	radio_fields = {'hist_diabetes': admin.HORIZONTAL}
	radio_fields = {'hist_': admin.HORIZONTAL}