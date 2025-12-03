<script setup>
import { computed } from 'vue'
import { FileText } from "@vicons/tabler"
import { NList, NListItem, NIcon, NThing, NText, NH3, NEllipsis } from 'naive-ui'

const props = defineProps({
  documents: {
    type: Array,
    required: true,
    default: () => []
  },
  selectedDocument: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['select-document'])

const isActive = (document) => {
  return props.selectedDocument && props.selectedDocument.id === document.id
}

const selectDocument = (document) => {
  emit('select-document', document)
}
</script>

<template>
  <div class="document-tree">
    <div class="tree-header">
      <n-h3 style="margin: 0;">Documents</n-h3>
    </div>
    
    <div class="tree-content">
      <n-list hoverable clickable>
        <n-list-item
          v-for="document in documents"
          :key="document.id"
          :class="{ 'active': isActive(document) }"
          @click="selectDocument(document)"
        >
          <template #prefix>
            <n-icon size="24" :depth="isActive(document) ? 1 : 3">
              <file-text />
            </n-icon>
          </template>
          
          <div class="document-item" style="min-width: 0; display: grid; gap: 4px;">
            <div class="document-title">
              <n-text :type="isActive(document) ? 'primary' : 'default'" strong>
                <n-ellipsis :tooltip="{ placement: 'right' }">{{ document.title }}</n-ellipsis>
              </n-text>
            </div>
            <div class="document-description" v-if="document.description">
              <n-text depth="3">
                <n-ellipsis :tooltip="{ placement: 'right' }">{{ document.description }}</n-ellipsis>
              </n-text>
            </div>
          </div>
        </n-list-item>
      </n-list>
    </div>
  </div>
</template>

<style scoped>
.document-tree {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.tree-header {
  padding: 1rem;
  border-bottom: 1px solid var(--n-border-color);
}

.tree-content {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.tree-content :deep(.n-list-item) {
  padding: 12px 16px;
  margin-bottom: 4px;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.tree-content :deep(.n-list-item:hover) {
  background-color: var(--n-color-hover);
}

.tree-content :deep(.n-list-item.active) {
  background-color: var(--n-color-primary-alpha1);
  border-left: 3px solid var(--n-color-primary);
}

.tree-content :deep(.n-list-item.active .n-thing-header__title) {
  color: var(--n-color-primary);
}

/* Scrollbar styling */
.tree-content::-webkit-scrollbar {
  width: 6px;
}

.tree-content::-webkit-scrollbar-track {
  background: var(--n-color-embedded);
  border-radius: 3px;
}

.tree-content::-webkit-scrollbar-thumb {
  background: var(--n-color-pressed);
  border-radius: 3px;
}

.tree-content::-webkit-scrollbar-thumb:hover {
  background: var(--n-color-disabled);
}

.document-title, .document-description {
  width: 100%;
  overflow: hidden;
}
</style>

<style>
.n-list-item__main, .n-thing-main {
  min-width: 0;
}
</style>