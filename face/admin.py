from django.contrib import admin
from .models import Travel_history,St_face
# Register your models here.
class Travel_historyAdmin(admin.ModelAdmin):
    list_display = ['student_id', 'history_3m']
admin.site.register(Travel_history, Travel_historyAdmin)
class faceadmin(admin.ModelAdmin):
    list_display = ['face']
admin.site.register(St_face, faceadmin)