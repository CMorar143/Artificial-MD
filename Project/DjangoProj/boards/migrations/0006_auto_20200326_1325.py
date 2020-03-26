# Generated by Django 2.2.7 on 2020-03-26 13:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0005_auto_20200319_1340'),
    ]

    operations = [
        migrations.AlterField(
            model_name='visit',
            name='date',
            field=models.DateTimeField(),
        ),
        migrations.CreateModel(
            name='Patient_Family',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('family', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Family')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
    ]
