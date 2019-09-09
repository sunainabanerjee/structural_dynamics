import os
import json
import coarse_graining as cg

if __name__ == "__main__":
    json_file = os.path.join(os.path.dirname(__file__), 'data', 'freesasa_out.json')
    area_info = cg.filtered_sasa([json_file],
                                 'ALA', area_type=[cg.SASAJsonFields().polar_area])
    print(json.dumps(area_info, indent=2))



