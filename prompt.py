from templates import *
import json

def generate_prompts_from_json(input_file,
                               prompt_template,
                               use_shorter_template=False,
                               current_utterance_only=False,
                               bart_summary=False,
                               has_persona_only=False,
                               has_knowledge_only=False,
                               has_persona_and_summary=False,
                               has_knowledge_and_summary=False,
                               no_prompt_blenderbot=False,
                               use_fsb_prompt=False,
                               segment_utt=False):
    def process_summary(history,summary):
        history_lines=history.split("\n")
        if history_lines[-1]=="__SILENCE__":
            last_line=history_lines[-2]
        else:
            last_line=history_lines[-1]
        new_summary=""
        if last_line.startswith("Person2:"):
            # new_summary=summary.replace("Person2","Bot").replace("Person1","User")
            pass
        else:
            # new_summary=summary.replace("Person2","User").replace("Person1","Bot")
            new_summary=summary.replace("Person2","Person1").replace("Person1","Person2")
        return new_summary
    
    def dfs_util(content,i,visited,new_content):
        #print(content)
        if i==len(content):
            return 

        visited[i]=True
        new_content.append(content[i])
        same_speaker=((i+1)<len(content)) and content[i].split(":")[0]==content[i+1].split(":")[0]
        #print(same_speaker)
        #print(len(content))
        #print(content[i].split(":")[0])
        #print(content[i+1].split(":")[0])
        if same_speaker and not visited[i+1]:
            dfs_util(content,i+1,visited,new_content)
    
    def dfs(content):
        content=content.split("\n")
        visited=[False]*len(content)
        revised_content=[]
        for i in range(len(content)):
            if not visited[i]:
                new_content=[]
                dfs_util(content,i,visited,new_content)
                #print(new_content)
                
                speaker=new_content[0].split(":")[0]
                utt=""
                prev_utt="."
                for j in range(len(new_content)):
                    if len(new_content[j])==0 or prev_utt==new_content[j]:
                        continue
                    utt+=new_content[j].split(":")[1]+" "
                    prev_utt=new_content[j]
                utt=utt.strip()
                utt=speaker+": "+utt
                if len(utt)>0:
                    revised_content.append(utt)

        return "\n".join(revised_content)
    
    with open(input_file,"r",encoding="utf-8") as fp:
         content=fp.readlines()

    summaries=[x.strip() for x in content]
    prompts=[]
    local_index=0
    for ind,entry in enumerate(summaries):
        if len(entry)==0 or ind==0 or len(summaries[ind-1])==0:
            continue
        entry=json.loads(entry)
        prev_entry=json.loads(summaries[ind-1])
        history=entry["history"]
        if "past_utterance" in entry:
        
            summary=entry["past_utterance"]
            prev_summary_lines=prev_entry["past_utterance"].split("\n")
        elif "semantic_utterances" in entry:
            summary=entry["semantic_utterances"]
            prev_summary=prev_entry["semantic_utterances"]
            if segment_utt:
                summary=dfs(summary)
                prev_summary=dfs(prev_summary)
                
            prev_summary_lines=prev_summary.split("\n")
        else:
            summary=entry["summary"].replace("<n>",".")
            prev_summary_lines=prev_entry["summary"].replace("<n>",".").split("\n")
        if bart_summary:
            #Convert Person1 and Person2 to User and Bot , based on the last line in history
            #check the last line of history
            summary=process_summary(entry["history"],entry["summary"])
            prev_summary=process_summary(prev_entry["history"],prev_entry["summary"])
            prev_summary_lines=prev_summary.split("\n")
            
        if has_persona_only or has_persona_and_summary or has_knowledge_only or has_knowledge_and_summary or use_fsb_prompt:
            if "user_summary" in entry:

                current_user_summary=entry["user_summary"].replace("\n",".")
                current_bot_summary=entry["bot_summary"].replace("\n",".")
                persona=current_user_summary+"\t"+current_bot_summary
                previous_user_summary=prev_entry["bot_summary"].replace("\n",".")
                previous_bot_summary=prev_entry["user_summary"].replace("\n",".")
            if "kg1_summary" in entry:
                kg1_summary=entry["kg1_summary"].replace("\n",".")
                kg2_summary=entry["kg2_summary"].replace("\n",".")
                kg3_summary=entry["kg3_summary"].replace("\n",".")
                total_knowledge="\t".join([kg1_summary,kg2_summary,kg3_summary])
                prev_kg1_summary=prev_entry["kg1_summary"].replace("\n",".") if "kg1_summary" in prev_entry else ""
                prev_kg2_summary=prev_entry["kg1_summary"].replace("\n",".") if "kg2_summary" in prev_entry else ""
                prev_kg3_summary=prev_entry["kg1_summary"].replace("\n",".") if "kg3_summary" in prev_entry else ""
                prev_total_knowledge="\t".join([prev_kg1_summary,prev_kg2_summary,prev_kg3_summary])
            
            
        current=entry["current_utterance"]
        response=entry["response"] if "response" in entry else entry["gold_response"]
        summary_lines=summary.split("\n")
        history_lines=history.split("\n")
        history_lines=[x for x in history_lines if not x.startswith("__SILENCE__")]
        last_two_utterances=history_lines[-2:]
        
        prev_utt=prev_entry["current_utterance"] 
        prev_response=prev_entry["response"] if "response" in prev_entry else prev_entry["gold_response"]
        new_sents=prev_summary_lines
        '''for i,sent in enumerate(prev_summary_lines):
            if bart_summary:
                new_sent=sent
            else:
                new_sent=str(i+1)+": "+sent
            new_sents.append(new_sent)'''
        prompt1="\n".join(new_sents)
        prompt2=prev_utt
        prompt3=prev_response
        '''new_sents=[]
        for i,sent in enumerate(summary_lines):
            if bart_summary or no_prompt_blenderbot:
                new_sent=sent
            else:
                new_sent=str(i+1)+": "+sent
            new_sents.append(new_sent)'''
        prompt4="\n".join(summary_lines)
        prompt5=current
        if use_shorter_template:
            if has_persona_and_summary:
                if prompt_template.count('{') != 4:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(current_user_summary,current_bot_summary,prompt4,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind,"personas":persona})
            elif has_knowledge_and_summary:                
                if prompt_template.count('{') != 3:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(total_knowledge,prompt4,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind})
            elif has_persona_only:
                if prompt_template.count('{') != 3:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(current_user_summary,current_bot_summary,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind,"personas":persona})
            elif has_knowledge_only:
                if prompt_template.count('{') != 2:
                    raise ValueError("The number of positional arguments is incorrect")
                
                prompts.append({"prompt":prompt_template.format(total_knowledge,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind})
                           
            
            elif current_utterance_only:
                #print(no_prompt_short_template)
                prompts.append({"prompt":prompt_template.format(prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"id":ind})
            elif no_prompt_blenderbot:
                prompts.append({"prompt":prompt5.strip(),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"id":ind})
              
            else:

                prompts.append({"prompt":prompt_template.format(prompt4,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind})
           
                
       
        else:
            if has_persona_and_summary:
                if prompt_template.count('{') != 9:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(previous_user_summary,previous_bot_summary,prompt1,prompt2,prompt3,current_user_summary,current_bot_summary,prompt4,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind,"personas":persona})
                
            elif has_knowledge_and_summary:
                if prompt_template.count('{') != 7:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(total_knowledge,prompt1,prompt2,prompt3,prev_total_knowledge,prompt4,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind})
                
            elif has_persona_only:
                if prompt_template.count('{') != 7:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(previous_user_summary,previous_bot_summary,prompt2,prompt3,current_user_summary,current_bot_summary,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind,"personas":persona})
            
            elif has_knowledge_only:
                if prompt_template.count('{') != 5:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(total_knowledge,prompt2,prompt3,prev_total_knowledge,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind})
                  
            elif current_utterance_only:
                if prompt_template.count('{') != 3:
                    raise ValueError("The number of positional arguments is incorrect")
                prompts.append({"prompt":prompt_template.format(prompt2,prompt3,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"id":ind})
            elif no_prompt_blenderbot:
                prompts.append({"prompt":prompt4+"\n"+prompt5.strip(),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4.strip(),"id":ind})
                         
            else:
                if prompt_template.count('{') != 5:
                    raise ValueError("The number of positional arguments is incorrect")
                 
                prompts.append({"prompt":prompt_template.format(prompt1,prompt2,prompt3,prompt4,prompt5),"gold_response":response,"history":history.strip(),"current_utterance":prompt5.strip(),"summary":prompt4,"id":ind})
           
                 
    return prompts
    

# Unit Test
if __name__ == "__main__":
    from tqdm import tqdm
    import pprint
    
    # use_shorter_template=False,
    # current_utterance_only=False,
    # bart_summary=False,
    # has_persona_only=False,
    # has_knowledge_only=False,
    # has_persona_and_summary=False,
    # has_knowledge_and_summary=False,
    # no_prompt_blenderbot=False,
    # use_fsb_prompt=False,
    # segment_utt=False):
    
    # 1    
    input_file="context_data/multi_session_chat/previous_utterances/msc_previous_utterances_last2.txt"
    prompts=generate_prompts_from_json(input_file, pegasusft_template_pplbased_tk_instruct, use_shorter_template=True)
    for k in tqdm(prompts):
        pass
    
    pprint.pprint(k)
    
    # 2
    input_file = "context_data/multi_session_chat/generated_summary/test_summary_pegasusft_dialogdata.txt"
    prompts=generate_prompts_from_json(input_file, pegasusft_template_pplbased_tk_instruct, use_shorter_template=True, bart_summary=True)
    for k in tqdm(prompts):
        pass
    
    pprint.pprint(k)
    