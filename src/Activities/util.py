from csv import DictReader
import sys

def round_latlon(lat, lon):
    if lat is None or lon is None:
        return (None, None)
    return (round(float(lat), 2), round(float(lon), 2))

class Getnghd():
    def __init__(self):
        self.bins_to_nghds = {}
        self.nghdName = {}
        for line in DictReader(open('../xls/point_map.csv')):
            self.bins_to_nghds[(float(line['lat']), float(line['lon']))] = line['nghd']
            if line['nghd'] not in self.nghdName:
                self.nghdName[line['nghd']] = 0

    def getnghd(self, coordinate):
        bin = round_latlon(coordinate[0], coordinate[1])
        if bin in self.bins_to_nghds:
            return self.bins_to_nghds[bin]
        else:
            return 'Outside Neeraj'
    def getnghdName(self):
        return self.nghdName.keys()
tt = Getnghd()
print tt.getnghd((40.4406, -79.9959))
print tt.getnghdName()
