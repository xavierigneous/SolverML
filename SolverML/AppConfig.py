#The program return the configuration details
import configparser
config = configparser.RawConfigParser()
config.read(r'app.config')

for eachsection in config.items():
    details_dict = dict(config.items(eachsection[0]))
    for ikey,ivalue in details_dict.items(): #Iterate each value from the section
        vars()[ikey] = ivalue
