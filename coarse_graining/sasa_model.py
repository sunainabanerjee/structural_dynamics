import os
import json

__version__ = "1.0"
__all__ = ['SASAJsonFields', 'load_freesasa_json', 'filtered_sasa']


class SASAJsonFields:
    @property
    def residue_name(self):
        return 'residue'

    @property
    def residue_id(self):
        return 'id'

    @property
    def total_area(self):
        return 'total'

    @property
    def polar_area(self):
        return 'polar'

    @property
    def apolar_area(self):
        return 'apolar'

    @property
    def all_fields(self):
        return [self.residue_name, self.residue_id,
                self.total_area, self.polar_area,
                self.apolar_area]

    @property
    def area_fields(self):
        return [self.total_area, self.polar_area, self.apolar_area]


def load_freesasa_json(json_file):
    assert os.path.isfile(json_file)
    with open(json_file, "r") as fp:
        data = json.load(fp)
    assert ('results' in data) and len(data['results']) > 0
    assert ('structure' in data['results'][0]) and len(data['results'][0]['structure']) > 0
    chain_data = data['results'][0]['structure'][0]['chains']
    results = dict()
    json_fields = SASAJsonFields()
    for chain in chain_data:
        residues = chain['residues']
        chain_id = chain['label']
        results[chain_id] = []
        for residue in residues:
            res_name = residue['name']
            res_id = residue['number']
            total_area = residue['area']['total']
            polar_area = residue['area']['polar']
            apolar_area = residue['area']['apolar']
            results[chain_id].append({json_fields.residue_name: res_name,
                                      json_fields.residue_id: res_id,
                                      json_fields.total_area: total_area,
                                      json_fields.polar_area: polar_area,
                                      json_fields.apolar_area: apolar_area})
    return results


def filtered_sasa(json_files, amino, area_type=None):
    assert isinstance(json_files, list)
    report = dict()
    json_fields = SASAJsonFields()
    if area_type is None:
        area_type = json_fields.area_fields
    assert isinstance(area_type, list)
    for area in area_type:
        assert area in json_fields.area_fields

    for i, filename in enumerate(json_files):
        results = load_freesasa_json(filename)
        file_data = dict()
        for chain in results.keys():
            chain_data = dict()
            for residue in results[chain]:
                if residue[json_fields.residue_name] == amino:
                    resid = residue[json_fields.residue_id]
                    chain_data[resid] = dict()
                    for area in area_type:
                        chain_data[resid][area] = residue[area]
            if len(chain_data) > 0:
                file_data[chain] = chain_data
        if len(file_data) > 0:
            report[i] = file_data
    return report

