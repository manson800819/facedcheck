from django.db import models

# Create your models here.
class Travel_history(models.Model):
    student_id = models.TextField(primary_key=True)
    history_3m = models.TextField()
    last_modify_date = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "Travel_history"


class St_face(models.Model):
    face = models.TextField(primary_key=True)
    student_who = models.ForeignKey(Travel_history,
                                 related_name='swho'
                                 )
    last_modify_date = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "St_face"