"""untitled3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from face.views import hello_view,facesteam,namesteam,facescreate,create_view,search_bar,takefinish
from django.conf.urls import url
from django.contrib import admin
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^stream/', facesteam, name='test_stream'),
    url(r'^cstream/', facescreate, name='creat_stream'),
    url(r'^name/', namesteam, name='name_stream'),
    url(r'^take/', takefinish, name='take'),
    url(r'^create/$', create_view, name='create'),
    url(r'^$', hello_view, name='home'),
    url(r'^create/search$',
        search_bar,
        name='search'),

]