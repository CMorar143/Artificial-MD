# Generated by Django 2.2.7 on 2020-03-04 00:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0011_auto_20200304_0045'),
    ]

    operations = [
        migrations.AddField(
            model_name='reminder',
            name='patient',
            field=models.ForeignKey(default=2, on_delete=django.db.models.deletion.CASCADE, to='boards.Patient'),
            preserve_default=False,
        ),
    ]