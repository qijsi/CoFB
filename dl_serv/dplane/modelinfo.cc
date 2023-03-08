#include <iostream>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include "modelinfo.h"


#define MAX_JSON_CONTENT_SIZE 65535

void parsejsonfile(std::string filename, struct model_info &minfo) {
    if (filename.empty()) {
    std::cerr<<"No json configuration file"<<std::endl;
    return ;
    }

    FILE *fp = fopen(filename.c_str(), "rb");
    if (fp == NULL) {
        std::cerr<<"Can't open json configuration file"<<std::endl;
        return ;
    }

    char tmpBuf[MAX_JSON_CONTENT_SIZE];
    rapidjson::FileReadStream rfstream(fp, tmpBuf, sizeof(tmpBuf));
    rapidjson::Document doc;
    doc.Parse(tmpBuf);
    fclose(fp);
    
    if (doc.HasParseError()) {
        std::cerr<<"Parse json configuration error"<<std::endl;
        return;
    }

#if 0
    if (doc["batch_dim"].GetInt()) minfo.batch_dim = true;
    else
      minfo.batch_dim = false;
#endif

    minfo.batch_dim = doc["batch_dim"].GetBool();
    std::cout<<"batch_dim: "<<minfo.batch_dim<<std::endl;

    minfo.name = doc["name"].GetString();
    minfo.path = doc["path"].GetString();
    minfo.version = doc["version"].GetInt();
    minfo.max_batch_size = doc["max_batch_size"].GetInt();
   
    rapidjson::Value &inputs = doc["inputs"];
    if (inputs.IsArray()) {
         struct model_io tmp_input;
        for (int i=0; i<inputs.Capacity(); i++) {
            rapidjson::Value &val = inputs[i];
            tmp_input.name = val["name"].GetString();
            tmp_input.data_type = val["data_type"].GetString();
            for(int j=0; j<val["dims"].Capacity();j++)
            tmp_input.dims.emplace_back(val["dims"][j].GetInt());
        }
        minfo.minput.emplace_back(std::move(tmp_input));
    }

    rapidjson::Value &outputs = doc["outputs"];
    if (outputs.IsArray()) {
        struct model_io tmp_output;
        for (int i=0; i<outputs.Capacity(); i++) {
            rapidjson::Value &val = outputs[i];
            tmp_output.name = val["name"].GetString();
            tmp_output.data_type = val["data_type"].GetString();
            for(int j=0; j<val["dims"].Capacity();j++)
            tmp_output.dims.emplace_back(val["dims"][j].GetInt());
        }
        minfo.moutput.emplace_back(std::move(tmp_output));
    }
}
